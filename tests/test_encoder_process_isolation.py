import subprocess
import sys


def test_embedding_entrypoint_does_not_load_xgboost_native_runtime():
    """Importing XGBoost first reproducibly crashes torch image inference on this Mac."""
    command = (
        "import sys; import derm_context.embeddings; "
        "assert 'xgboost' not in sys.modules, 'Encoder process loaded XGBoost'"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
