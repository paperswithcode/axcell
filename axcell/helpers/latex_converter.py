#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import docker
from docker.errors import ContainerError, ImageNotFound
from pathlib import Path
from tempfile import TemporaryDirectory
from bs4 import BeautifulSoup

from axcell.errors import LatexConversionError


def ro_bind(path): return dict(bind=path, mode='ro')


def rw_bind(path): return dict(bind=path, mode='rw')


# magic constant used in latex2html.sh to signal that
# conversion failed on LaTeXML step
MAGIC_EXIT_ERROR = 117


class LatexConverter:
    def __init__(self, scripts_path=None):
        # pull arxivvanity/engrafo image
        self.client = docker.from_env()
        if scripts_path is None:
            self._scripts_path =\
                Path(__file__).resolve().parent.parent / 'scripts'
        else:
            self._scripts_path = Path(scripts_path)

    def latex2html(self, source_dir, output_dir, use_named_volumes=False):
        base = self._scripts_path
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        scriptname = "/files/latex2html.sh"
        filename = "index.html"

        volumes = {
            base / "latex2html.sh": ro_bind("/files/latex2html.sh"),
            base / "guess_main.py": ro_bind("/files/guess_main.py"),  # todo: run guess_main outside of docker
            base / "patches": ro_bind("/files/patches")   # todo: see which patches can be dropped
        }

        # In case of fully dockerized pipeline we use named volumes to share files between the steps.
        # This, however, requires as to mount specific volumes with all papers, not only the currently processed one.
        # (see https://github.com/moby/moby/issues/32582)
        if use_named_volumes:
            volumes.update({
                "pwc_unpacked_sources": ro_bind("/data/arxiv/unpacked_sources"),
                "pwc_htmls": rw_bind("/data/arxiv/htmls")
            })
            command = [scriptname, filename, str(source_dir), str(output_dir)]
        else:
            volumes.update({
                source_dir.resolve(): ro_bind("/files/ro-source"),
                output_dir.resolve(): rw_bind("/files/htmls")
            })
            command = [scriptname, filename]

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.client.containers.run("arxivvanity/engrafo:b3db888fefa118eacf4f13566204b68ce100b3a6", command, remove=True, volumes=volumes)
        except ContainerError as err:
            if err.exit_status == MAGIC_EXIT_ERROR:
                raise LatexConversionError("LaTeXML was unable to convert source code of this paper")
            if "Unable to find any suitable tex file" in err.stderr.decode('utf-8'):
                raise LatexConversionError("Unable to find any suitable tex file")
            raise

    # todo: check for errors
    def clean_html(self, path):
        path = Path(path)
        with path.open("rb") as file:
            soup = BeautifulSoup(file, "html5lib")
        return str(soup)

    def to_html(self, source_dir):
        with TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)
            try:
                self.latex2html(source_dir, output_dir)
                return self.clean_html(output_dir / "index.html")
            except ContainerError as err:
                raise LatexConversionError from err
