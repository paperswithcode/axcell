import logging
from pathlib import  Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARN)

# used only to dynamically fetch graph ql data
graphql_url = 'http://10.0.1.145:8001/graphql/'

# otherwise use this files
data = Path("/mnt/efs/pwc/data")
goldtags_dump = data / "dumps" / "goldtags-2019.07.31_1454-htlatex-latexml.json.gz"


elastic = dict(hosts=['localhost'], timeout=20)


arxiv = data/'arxiv'
htmls_raw = arxiv/'htmls'
htmls_clean = arxiv/'htmls-clean'

datasets = data/"datasets"
datasets_structure = datasets/"structure"
