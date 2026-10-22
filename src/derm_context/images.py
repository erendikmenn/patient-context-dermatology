"""Read images without extracting the 400k-file archive; ready for encoder batches."""

import io
import zipfile
from pathlib import Path

from PIL import Image


class ImageArchive:
    def __init__(self, path: Path):
        self.archive = zipfile.ZipFile(path)

    def read(self, member: str) -> Image.Image:
        with Image.open(io.BytesIO(self.archive.read(member))) as image:
            return image.convert("RGB")

    def close(self):
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
