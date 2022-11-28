#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from magic import Magic
import tarfile
import gzip
from pathlib import Path
from shutil import copyfileobj
from axcell.errors import UnpackError
from ..pipeline_logger import pipeline_logger


class Unpack:
    step = "unpack"

    def __init__(self):
        self.magic = Magic(mime=True, uncompress=True)
        self.magic_formatted = Magic(mime=False, uncompress=True)

    def __call__(self, source, dest):
        pipeline_logger(f"{Unpack.step}::call", source=source, dest=dest)
        source = Path(source)
        dest = Path(dest)
        mime = self.magic.from_file(str(source))
        pipeline_logger(f"{Unpack.step}::detect_mime", source=source, mime=mime)
        if mime == 'application/x-tar':
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(source, "r:*") as tar:
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, dest)
        elif mime == 'text/x-tex':
            dest.mkdir(parents=True, exist_ok=True)
            with gzip.open(source, "rb") as src, open(dest / "main.tex", "wb") as dst:
                copyfileobj(src, dst)
        elif mime == 'application/pdf':
            raise UnpackError(f"No LaTeX source code available for this paper, PDF only")
        elif mime == 'text/plain' and 'withdrawn' in self.magic_formatted.from_file(str(source)):
            raise UnpackError(f"The paper has been withdrawn and there is"
                              f" no LaTeX source code available")
        else:
            raise UnpackError(f"Cannot unpack file of type {mime}")
