from magic import Magic
import tarfile
import gzip
from pathlib import Path
from shutil import copyfileobj
from sota_extractor2.errors import UnpackError
from ..pipeline_logger import pipeline_logger


class Unpack:
    step = "unpack"

    def __init__(self):
        self.magic = Magic(mime=True, uncompress=True)

    def __call__(self, source, dest):
        pipeline_logger(f"{Unpack.step}::call", source=source, dest=dest)
        source = Path(source)
        dest = Path(dest)
        mime = self.magic.from_file(str(source))
        pipeline_logger(f"{Unpack.step}::detect_mime", source=source, mime=mime)
        if mime == 'application/x-tar':
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(source, "r:*") as tar:
                tar.extractall(dest)
        elif mime == 'text/x-tex':
            dest.mkdir(parents=True, exist_ok=True)
            with gzip.open(source, "rb") as src, open(dest / "main.tex", "wb") as dst:
                copyfileobj(src, dst)
        elif mime == 'application/pdf':
            raise UnpackError(f"No LaTeX source code available for this paper, PDF only")
        else:
            raise UnpackError(f"Cannot unpack file of type {mime}")
