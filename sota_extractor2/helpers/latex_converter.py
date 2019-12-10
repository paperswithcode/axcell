import docker
from docker.errors import ContainerError, ImageNotFound
from pathlib import Path
from tempfile import TemporaryDirectory

from sota_extractor2.errors import LatexConversionError


def ro_bind(path): return dict(bind=path, mode='ro')


def rw_bind(path): return dict(bind=path, mode='rw')


class LatexConverter:
    def __init__(self, base_path):
        # pull arxivvanity/engrafo image
        self.client = docker.from_env()
        self.base_path = Path(base_path)

    def latex2html(self, source_dir, output_dir, use_named_volumes=False):
        base = self.base_path
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
        self.client.containers.run("arxivvanity/engrafo:b3db888fefa118eacf4f13566204b68ce100b3a6", command, remove=True, volumes=volumes)

    # todo: check for errors

    def clean_html(self, path, use_named_volumes=False):
        path = Path(path)

        if use_named_volumes:
            index_path = path
            volumes = {
                "pwc_htmls": ro_bind("/data/arxiv/htmls")
            }
        else:
            index_path = "/files/index.html"
            volumes = {
                path.resolve(): ro_bind(index_path)
            }

        command = ["timeout", "-s", "KILL", "20", "chromium-browser", "--headless",
                   "--disable-gpu", "--disable-software-rasterizer", "--no-sandbox",
                   "--timeout=30000", "--dump-dom", str(index_path)]
        data = self.client.containers.run("zenika/alpine-chrome:73", command, remove=True, entrypoint="",
                                          volumes=volumes)
        return data.decode('utf-8')

    def to_html(self, source_dir):
        with TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)
            try:
                self.latex2html(source_dir, output_dir)
                return self.clean_html(output_dir / "index.html")
            except ContainerError as err:
                raise LatexConversionError from err
