import numpy as np
import pandas as pd
import pytest

from derm_context.features import feature_columns, make_preprocessor, select_features
from derm_context.prepare import assert_split_integrity, merge_labels, split_patients


def cohort():
    return pd.DataFrame(
        [
            {
                "isic_id": f"image_{patient}_{lesion}",
                "patient_id": f"p{patient:03d}",
                "target": int(patient < 30 and lesion == 0),
            }
            for patient in range(100)
            for lesion in range(3)
        ]
    )


def test_splits_are_patient_disjoint_deterministic_and_complete():
    frame = cohort()
    frame["split"] = split_patients(frame)
    assert frame.groupby("patient_id")["split"].nunique().max() == 1
    assert set(frame["split"]) == {"train", "validation", "test"}
    pd.testing.assert_series_equal(frame["split"], split_patients(frame), check_names=False)
    shuffled = frame.sample(frac=1, random_state=18).reset_index(drop=True)
    shuffled["split"] = split_patients(shuffled)
    assert (
        frame.set_index("isic_id")["split"].to_dict()
        == shuffled.set_index("isic_id")["split"].to_dict()
    )


def test_patient_overlap_is_rejected():
    frame = cohort()
    frame["split"] = split_patients(frame)
    first = frame.loc[0, "split"]
    frame.loc[1, "split"] = "test" if first != "test" else "train"
    with pytest.raises(ValueError, match="overlap"):
        assert_split_integrity(frame)


@pytest.mark.parametrize("failure", ["duplicate", "missing_id", "missing_patient", "bad_label"])
def test_bad_inputs_fail_closed(failure):
    frame = cohort()
    labels = frame[["isic_id", "target"]].rename(columns={"target": "malignant"})
    metadata = frame.drop(columns="target")
    if failure == "duplicate":
        labels = pd.concat([labels, labels.iloc[[0]]])
    elif failure == "missing_id":
        labels = labels.iloc[1:]
    elif failure == "missing_patient":
        metadata.loc[0, "patient_id"] = None
    else:
        labels.loc[0, "malignant"] = -999
    with pytest.raises(ValueError):
        merge_labels(metadata, labels)


def test_features_ignore_outcomes_ids_and_external_classifiers():
    numeric, categorical = feature_columns("morphometry")
    frame = pd.DataFrame({c: [1.0, 2.0] for c in numeric})
    for c in categorical:
        frame[c] = ["known", None]
    for c in [
        "target",
        "patient_id",
        "iddx_full",
        "mel_thick_mm",
        "attribution",
        "tbp_lv_nevi_confidence",
        "tbp_lv_dnn_lesion_confidence",
    ]:
        frame[c] = [999, 999]
    selected = select_features(frame, "morphometry")
    assert set(selected) == set(numeric + categorical)
    preprocessor = make_preprocessor("morphometry")
    preprocessor.fit(selected)
    unseen = selected.copy()
    unseen["sex"] = "new category only in validation"
    assert np.isfinite(preprocessor.transform(unseen)).all()


def test_missing_feature_fails_instead_of_silently_changing_model():
    with pytest.raises(ValueError, match="Missing required"):
        select_features(pd.DataFrame({"age_approx": [40]}), "clinical")
