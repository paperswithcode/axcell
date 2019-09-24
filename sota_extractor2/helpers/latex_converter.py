import docker
from pathlib import Path

def ro_bind(path): return dict(bind=path, mode='ro')
def rw_bind(path): return dict(bind=path, mode='rw')



class LatexConverter:
    def __init__(self, base_path):
        # pull arxivvanity/engrafo image
        self.client = docker.from_env()
        self.base_path = Path(base_path)

    def to_html(self, source_dir, output_dir):
        base = self.base_path
        volumes = {
            base / "latex2html.sh": ro_bind("/files/latex2html.sh"),
            base / "guess_main.py": ro_bind("/files/guess_main.py"),  # todo: run guess_main outside of docker
            base / "patches": ro_bind("/files/patches"),  # todo: see which patches can be dropped
            source_dir.resolve(): ro_bind("/files/ro-source"),
            output_dir.resolve(): rw_bind("/files/htmls")
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "index.html"
        command = ["/files/latex2html.sh", filename]
        self.client.containers.run("arxivvanity/engrafo", command, remove=True,
                              volumes=volumes)  # todo: check if command as a list protects from shell injection
    # todo: check for errors