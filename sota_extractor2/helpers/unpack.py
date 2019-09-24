from magic import Magic
import tarfile
import gzip
from pathlib import Path
from shutil import copyfileobj
from sota_extractor2.errors import UnpackError


class Unpack:
    def __init__(self):
        self.magic = Magic(mime=True, uncompress=True)

    def __call__(self, source, dest):
        source = Path(source)
        dest = Path(dest)
        mime = self.magic.from_file(str(source))
        if mime == 'application/x-tar':
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(source, "r:*") as tar:
                tar.extractall(dest)
        elif mime == 'text/x-tex':
            dest.mkdir(parents=True, exist_ok=True)
            with gzip.open(source, "rb") as src, open(dest / "main.tex") as dst:
                copyfileobj(src, dst)
        else:
            raise UnpackError(f"Cannot unpack file of type {mime}")
