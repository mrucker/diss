import pandas as pd
import shutil

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

maxworkers = 12

def write_file(old_path):
    new_path = Path(str(old_path).replace("/old/","/new/"))
    if new_path.exists(): return 1
    print(new_path)
    new_path.parent.mkdir(parents=True,exist_ok=True)
    pd.read_csv(old_path).sort_values("TimestampUtc").to_csv(new_path,index=False)
    return new_path

shutil.rmtree("./sapiens/data/phone/new",ignore_errors=True)
with ProcessPoolExecutor(max_workers=maxworkers) as executor:
    list(executor.map(write_file,list(Path("./sapiens/data/phone").glob("**/*.csv"))))