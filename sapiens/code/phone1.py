import csv
import gzip
import time
import pytz
import traceback
import orjson as json
import pandas as pd
import shutil

from pathlib import Path
from datetime import datetime
from operator import itemgetter
from itertools import groupby, count, islice
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Optional

maxworkers = 20

prd = lambda k: k[:8] >= '20231001' and ('8am' in k or '10am' in k)
src = "/mnt/sda/sapiens_phone/data/"
dst = "./sapiens/data/phone/old"

write = True

keepers = {"Accelerometer", "Battery", "LinearAcceleration", "Proximity", "Wlan", "Speed", "Pedometer", "Location"}
drop    = {'Probe', 'DeviceManufacturer', 'SensingAgentStateDescription', 'TaggedEventTags', 'Id', 'DeviceModel', 'TaggedEventId', 'ProtocolId', 'LocalOffsetFromUTC', 'BuildId', 'OperatingSystem','IncompleteDatum', 'TriggerDatumId', "ParticipantId", "LocalTimeZone","DeviceId"}
order   = ['TimestampUtc']

def get_files(predicate=(lambda _:True)):
    _src = src.rstrip('/')
    keys = [str(d).replace('\\','/') for d in Path(_src).glob("**/*.gz")]
    return [key for key in keys if predicate(key[len(_src)+1:])]

def get_stamp_utc(stamp:str) -> Optional[str]:
    return get_stamp_utc_with_tz(stamp)[0]

def get_stamp_utc_with_tz(stamp:str) -> Tuple[Optional[str],Optional[str]]:

    if not stamp:
        return (None, None)

    has_tz   = (stamp[-6] in ['-','+'] or stamp[-5] in ['-','+']) and stamp[10] in ['T',' ']
    tz_start = max(stamp.rfind('+'),stamp.rfind('-')) if has_tz else None
    tz       = stamp[tz_start:] if has_tz else "+00:00"

    try:
        dt = datetime.fromisoformat(stamp).astimezone(pytz.utc)
    except ValueError:

        dt = stamp[:tz_start]

        if tz[-3] != ':': tz = f"{tz[:-3]}:{tz[-3:]}"
        if '/' in dt: dt = dt.replace('/','-')
        if "." in dt: dt = dt.ljust(26,'0')

        try:
            dt = datetime.fromisoformat(f"{dt}{tz}").astimezone(pytz.utc)
        except ValueError:
            return (None, None)

    return f'{dt.timestamp():.3f}', tz

def get_probe(metadata:str):
    metadata    = metadata or ''
    first_comma = metadata.find(',')
    last_period = metadata.rfind('.',None,first_comma)
    return metadata[last_period+1:first_comma].removesuffix('Datum')

def get_single(x):
    if isinstance(x,pd.Series):
        x = x.dropna().drop_duplicates()
        return x if len(x) == 1 else list(set(x))
    return x[0] if len(x)==1 else x

def get_headers(rows,order:list,drop:set):
    full_keys = set.union(*[set(i.keys()) for i in rows])
    return order+sorted(full_keys-drop-set(order))

def clean_script(rows):
    rows = sorted(rows,key=itemgetter('GroupId','SubmissionTimestampUtc'))
    grps = [list(g) for _,g in groupby(rows,key=itemgetter('GroupId','SubmissionTimestampUtc'))]
    grps = sorted(grps,key=lambda grp: min(map(itemgetter('TimestampUtc'),grp)))

    #This step is here for an old bug that appears to have been fixed in February 2023
    is_input = lambda row: 'Text' != row['InputName']
    is_text  = lambda row: 'Text' == row['InputName']

    rows = []
    for group in grps:
        text = next(filter(is_text ,group),None)
        if any(map(is_input,group)):
            for input in filter(is_input,group):
                input['Text'] = text['Response'] if text else '-'
                if 'Slider' in input['InputName']: input['InputName'] = '-'
                rows.append(input)
        else:
            for text in filter(is_text,group):
                text['Text'] = text['Response']
                text['Response' ] = '-'
                text['InputName'] = '-'
                rows.append(text)

    for row in rows:
        row['InputName'] = str(row['InputName']).replace('\n', ' ')
        row['Text'     ] = str(row['Text'     ]).replace('\n', ' ')
        row['Response' ] = str(row['Response' ]).replace('\n', ' ')

    return rows

def open_file(key:str):
    return (gzip.open if ".gz" in str(key) else open)(key, 'rt')

def read_json(lines:str):
    try:
        json_obj = json.loads(lines)
        if isinstance(json_obj,list):
            yield from json_obj
        else:
            yield json_obj
    except Exception:
        for line in lines.splitlines():
            line = line.strip().strip(',')
            if line:
                if line[0] + line[-1] not in ['[]','{}']: continue
                try:
                    if line.startswith("["): yield from json.loads(line)
                    if line.startswith("{"): yield json.loads(line)
                except Exception:
                    pass

def read_file(key:str) -> dict:
    datums       = []
    file_context = open_file(key)
    pid          = key.split("/")[-2][-3:]

    if file_context is None:
        return {}

    with file_context as h:
        try:
            for o in read_json(h.read()):

                _type = o.pop("$type",None)
                if not _type: continue

                _pid = o.get('ParticipantId') or pid
                if not _pid: continue

                _probe = get_probe(_type)
                if _probe not in keepers: continue

                _timestamp = o.pop('Timestamp',None)
                if not _timestamp: continue

                if 'IncompleteDatum' in o and o['IncompleteDatum'] == True:
                    print("INCOMPLETE")
                    continue

                o["Probe"        ] = _probe
                o["TimestampUtc" ] = get_stamp_utc(_timestamp)

                datums.append(o)

        except UnicodeDecodeError:
            pass #file is corrupt

    out = {}
    
    for probe, group in groupby(sorted(datums,key=itemgetter('Probe')),key=itemgetter('Probe')):
        group = list(group)

        if group:
            headers  = get_headers(group,order,drop)

            rows   = [ list(map(row.get,headers)) for row in group ]
            floats = [ h for h in range(len(headers)) if isinstance(rows[0][h],float)]

            if floats:
                for row in rows:
                    for h in floats:
                        if isinstance(row[h],float):
                            row[h] = f"{row[h]:g}" if row[h].is_integer() else f"{row[h]:.7f}"

            out[(pid,probe)] = [headers, *rows]

    return out

def write_rows(fobj, rows) -> None:
    rows    = iter(rows)
    headers = next(rows)

    writer = csv.writer(fobj,lineterminator='\n',strict=True)
    if fobj.tell() == 0: writer.writerow(headers)
    writer.writerows(rows)

def write_scripts(path):
    if Path(path).exists():
        data = pd.read_csv(path,keep_default_na=False)

        for name, rows in data.groupby('ScriptName'):
            name = str(name).replace("?","").replace(" ", "_")

            row_order = ["SubmissionTimestampUtc","LocalTimeZone","ParticipantId","DeviceId","RunId"]
            col_order = ["SubmissionTimestampUtc","LocalTimeZone","ParticipantId","DeviceId","RunId"]

            rows = rows[rows['InputName'] != '-']

            if len(rows) > 0:
                rows = rows.pivot_table(index=row_order,values='Response',columns='InputName',aggfunc=get_single).reset_index()
                rows = rows.fillna("-")
                cols = col_order+list(rows.columns)[len(row_order):]
                rows[cols].to_csv(f'{dst}/Script_{name}.csv',sep=',',encoding='utf-8',index=False)

if __name__ == '__main__':

    t1 = 0
    t2 = 0
    t3 = 0

    if write:
        shutil.rmtree(dst,ignore_errors=True)
        Path(dst).mkdir(parents=True,exist_ok=True)
        

    def get_results(files):
        if maxworkers == 1:
            yield from iter(map(read_file,files))
        else:
            files = iter(files)
            with ProcessPoolExecutor(max_workers=maxworkers) as executor:
                while chunk := list(islice(files,500)):
                    yield from iter(executor.map(read_file,chunk))

    fobjs   = {}
    files   = get_files(prd)
    results = get_results(files)
    
    try:
        for i in count(1):
            try:
                s3 = time.time()

                s1 = time.time()
                datums = next(results)
                t1 += time.time()-s1

                for (pid,probe),rows in datums.items():
                    if write:
                        out = f"{dst}/{pid}"
                        Path(out).mkdir(parents=True,exist_ok=True)
                        probe = f"{out}/{probe}"
                        if probe not in fobjs: fobjs[probe] = open(f"{probe}.csv",mode='w',newline='')
                        s2 = time.time()
                        write_rows(fobjs[probe],rows)
                        t2 += time.time()-s2

                t3 += time.time()-s3
                print(f"{t1/i:0<6.4f} {t2/i:0<6.4f} {t3/i:0<6.4f} File {i}/{len(files)}")

            except StopIteration:
                raise
            except Exception:
                print(''.join(traceback.format_exc()))

    except StopIteration:
        pass
    finally:
        for f in fobjs.values(): f.close()