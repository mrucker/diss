import pickle
import pandas as pd
from pathlib import Path

for p in Path("/mnt/sda/sapiens_watch/").iterdir():
    pid = int(p.name[1:])
    print(pid)
    Path(f"./sapiens/data/watch/{pid}").mkdir(parents=True,exist_ok=True)

    if not (p/"Smartwatch_HeartRateDatum.pkl").exists():
        print("A")
    elif not Path(f"./sapiens/data/watch/{pid}/HeartRate.csv").exists():
        try:
            obj = pickle.loads((p/"Smartwatch_HeartRateDatum.pkl").read_bytes())
            obj = obj[["T","HR"]].sort_values("T")
            obj["T"] = obj["T"].dt.tz_convert("UTC")
            obj["TimestampUtc"] = ((obj["T"] - pd.Timestamp.utcfromtimestamp(0)) / pd.Timedelta("1s")).round(3)
            obj[["TimestampUtc","HR"]].to_csv(f"./sapiens/data/watch/{pid}/HeartRate.csv",index=False)
        except Exception as e:
            print("HR")

    if not (p/"Smartwatch_LinearAccelerationDatum.pkl").exists():
        print("B")
    elif not Path(f"./sapiens/data/watch/{pid}/LinearAcceleration.csv").exists():
        try:
            obj = pickle.loads((p/"Smartwatch_LinearAccelerationDatum.pkl").read_bytes())
            obj = obj[["T","X","Y","Z"]].sort_values("T")
            obj["T"] = obj["T"].dt.tz_convert("UTC")
            obj["TimestampUtc"] = ((obj["T"] - pd.Timestamp.utcfromtimestamp(0)) / pd.Timedelta("1s")).round(3)
            obj[["TimestampUtc","X","Y","Z"]].to_csv(f"./sapiens/data/watch/{pid}/LinearAcceleration.csv",index=False)
        except Exception as e:
            print("LA")

    if not (p/"Smartwatch_PPGDatum.pkl").exists():
        print("C")
    elif not Path(f"./sapiens/data/watch/{pid}/PPG.csv").exists():
        try:
            obj = pickle.loads((p/"Smartwatch_PPGDatum.pkl").read_bytes())
            obj = obj[["T"]+[f"PPG{i}" for i in range(1,17)]].sort_values("T")
            obj["T"] = obj["T"].dt.tz_convert("UTC")

            for i in range(1,17): obj[f"PPG{i}"] = pd.to_numeric(obj[f"PPG{i}"],errors="coerce")

            #I don't know how I feel about this... without it though
            #these files are HUGE, with it they are just large
            obj = obj.set_index("T")
            obj = obj.resample(pd.Timedelta(seconds=.1)).mean()
            obj = obj.reset_index()

            obj["TimestampUtc"] = ((obj["T"] - pd.Timestamp.utcfromtimestamp(0)) / pd.Timedelta("1s")).round(3)
            obj[["TimestampUtc"]+[f"PPG{i}" for i in range(1,17)]].to_csv(f"./sapiens/data/watch/{pid}/PPG.csv",index=False)
        except Exception as e:
            print(str(e))
            print("PPG")

    if not (p/"Smartwatch_StepCountDatum.pkl").exists():
        print("D")
    elif not Path(f"./sapiens/data/watch/{pid}/StepCount.csv").exists():
        try:
            obj = pickle.loads((p/"Smartwatch_StepCountDatum.pkl").read_bytes())
            obj = obj[["T","Count"]].sort_values("T")
            obj["T"] = obj["T"].dt.tz_convert("UTC")
            obj["TimestampUtc"] = ((obj["T"] - pd.Timestamp.utcfromtimestamp(0)) / pd.Timedelta("1s")).round(3)
            obj[["TimestampUtc","Count"]].to_csv(f"./sapiens/data/watch/{pid}/StepCount.csv",index=False)
        except Exception as e:
            print("SC")
