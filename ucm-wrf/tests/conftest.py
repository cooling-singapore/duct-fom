import os.path

from dotenv import load_dotenv
import shutil

load_dotenv()

os.environ['MODEL_PATH'] = os.path.join(os.getcwd(), '..', 'model')
os.environ['USE_NETCDF_VERSION'] = "4"

def copy_files(mappings: list) -> None:
    for item in mappings:
        shutil.copyfile(item['from'], item['to'])

