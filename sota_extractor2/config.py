import logging
from pathlib import  Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARN)

# used only to dynamically fetch graph ql data
graphql_url = 'http://10.0.1.145:8001/graphql/'

# otherwise use this files
data = Path("/mnt/efs/pwc/data")
goldtags_dump = data / "dumps" / "goldtags-2019.09.13_0219.json.gz"


elastic = dict(hosts=['localhost'], timeout=20)


arxiv = data/'arxiv'
htmls_raw = arxiv/'htmls'
htmls_clean = arxiv/'htmls-clean'

datasets = data/"datasets"
datasets_structure = datasets/"structure"
structure_models = datasets / "structure" / "models"

mocks = datasets / "mocks"

linking_models = datasets / "linking" / "models"
linking_data = datasets / "linking" / "data"
