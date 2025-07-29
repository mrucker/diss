#Setup Code

import csv
import warnings
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from itertools import islice, chain, count, product
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor

data_dir = "./sapiens/data"

import torch
import torch.utils.data
import coba as cb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from parameterfree import COCOB
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

try:
    torch.set_num_threads(3)
    torch.set_num_interop_threads(3)
except RuntimeError:
    pass

c0 = "#444"
c1 = "#0072B2"
c2 = "#E69F00"
c3 = "#009E73"
c4 = "#56B4E9"
c5 = "#D55E00"
c6 = "#F0E442"
c7 = "#CC79A7"
c8 = "#000000"
c9 = "#332288"

torch.set_default_device('cpu')
plt.rc('font', **{'size': 20})

def make_emotions_df():

    def add_day_columns(df, timestamp_col, participant_df):
        return add_rel_day(add_start_day(add_day(df, timestamp_col), participant_df))

    def add_day(df, timestamp_col):
        df = df.copy()
        df["Day"] = (df[timestamp_col]/(60*60*24)).apply(np.floor)
        return df

    def add_start_day(df, participant_df):
        participant_df = participant_df.copy()
        participant_df["StartDay"] = (participant_df["DataStartStampUtc"]/(60*60*24)).apply(np.floor)
        return pd.merge(df, participant_df[['ParticipantId',"StartDay"]])

    def add_rel_day(df):
        df = df.copy()
        df["RelDay"] = df["Day"]-df["StartDay"]
        return df

    def drop_all1_ends(df):
        last, last_gt_1, keep = df.copy(),df.copy(), df.copy()
        
        last_gt_1 = last_gt_1[last_gt_1["State Anxiety"]!= 1]
        last_gt_1 = last_gt_1.groupby("ParticipantId")["RelDay"].max().reset_index()
        last_gt_1 = last_gt_1.rename(columns={"RelDay":"Last Day > 1"})

        last = last.groupby("ParticipantId")["RelDay"].max().reset_index()
        last = last.rename(columns={"RelDay":"Last Day"})

        for pid in last["ParticipantId"]:
            
            last_day = last[last["ParticipantId"]==pid]["Last Day"].item()
            last_day_gt_1 = last_gt_1[last_gt_1["ParticipantId"]==pid]["Last Day > 1"].item()
            
            if last_day-last_day_gt_1 >= 3:
                is_not_pid = keep["ParticipantId"] != pid
                is_lt_day  = keep["RelDay"] <= last_day_gt_1
                keep = keep[is_not_pid | is_lt_day]

        return keep

    emotions_df = pd.read_csv(f'{data_dir}/Emotions.csv')
    participant_df = pd.read_csv(f'{data_dir}/Participants.csv')

    emotions_df = emotions_df[emotions_df["WatchDataQuality"] == "Good"]

    emotions_df["State Anxiety"] = pd.to_numeric(emotions_df["State Anxiety"], errors='coerce')
    emotions_df["ER Interest"] = pd.to_numeric(emotions_df["ER Interest"], errors='coerce')
    emotions_df["Phone ER Interest"] = pd.to_numeric(emotions_df["Phone ER Interest"], errors='coerce')
    emotions_df["Response Time (min)"] = (emotions_df["SubmissionTimestampUtc"] - emotions_df["DeliveredTimestampUtc"])/60
    emotions_df["Response Time (sec)"] = (emotions_df["SubmissionTimestampUtc"] - emotions_df["DeliveredTimestampUtc"])
    emotions_df["Response Time (log min)"] = np.log((1+ emotions_df["SubmissionTimestampUtc"] - emotions_df["DeliveredTimestampUtc"])/60)
    emotions_df["Response Time (log sec)"] = np.log((1+ emotions_df["SubmissionTimestampUtc"] - emotions_df["DeliveredTimestampUtc"]))

    emotions_df["State Anxiety (z)"] = float('nan')
    emotions_df["ER Interest (z)"] = float('nan')

    for pid in set(emotions_df["ParticipantId"].tolist()):
        is_pid = emotions_df["ParticipantId"] == pid
        is_anx = emotions_df["State Anxiety"] > 1
        emotions_df.loc[is_pid,["ER Interest (z)"]] = StandardScaler().fit_transform(emotions_df.loc[is_pid,["ER Interest"]])
        emotions_df.loc[is_pid&is_anx,["State Anxiety (z)"]] = StandardScaler().fit_transform(emotions_df.loc[is_pid&is_anx,["State Anxiety"]])

    emotions_df = add_day_columns(emotions_df, "DeliveredTimestampUtc", participant_df)

    emotions_df = emotions_df[emotions_df["RelDay"] < 11]

    return drop_all1_ends(emotions_df)

emotions_df = make_emotions_df()

class TheoryGridCellSpatialRelationEncoder:
    #https://arxiv.org/pdf/2003.00824
    def __init__(self, coord_dim = 2, frequency_num = 16, max_radius = 10000,  min_radius = 1000, freq_init = "geometric"):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """

        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init

        # the frequency we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis = 1)

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])                        # 0
        self.unit_vec2 = np.asarray([-1.0/2.0, np.sqrt(3)/2.0])      # 120 degree
        self.unit_vec3 = np.asarray([-1.0/2.0, -np.sqrt(3)/2.0])     # 240 degree

        # compute the dimention of the encoded spatial relation embedding
        self.input_embed_dim = int(6 * self.frequency_num)
        
    def cal_freq_list(self):
        if self.freq_init == "random":
            self.freq_list = np.random.random(size=[self.frequency_num]) * self.max_radius
        elif self.freq_init == "geometric":
            log_timescale_increment = (np.log(float(self.max_radius) / float(self.min_radius)) /(self.frequency_num*1.0 - 1))
            timescales = self.min_radius * np.exp(np.arange(self.frequency_num).astype(float) * log_timescale_increment)
            self.freq_list = 1.0/timescales
        else:
            raise Exception()

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            coords = [[c] for c in coords]
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec 
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis = -1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis = -1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1
        
        return spr_embeds.squeeze().tolist()

def is_gt(values,val):
    out = (values > val).astype(float)
    out[values.isna()] = float('nan')
    return out

def is_lt(values,val):
    out = (values < val).astype(float)
    out[values.isna()] = float('nan')
    return out

def scale(values):
    with warnings.catch_warnings():
        # If a column has all nan then a warning is thrown
        # We supress that warning because that can happen
        warnings.simplefilter("ignore")
        return StandardScaler().fit_transform(values).tolist()

def add1(X):
    for x,z in zip(X,np.isnan(X).all(axis=1).astype(int)):
        x.append(z)
    return X

def wins(file_path, timestamps, window_len):
    file = open(file_path) if Path(file_path).exists() else nullcontext()
    rows = islice(csv.reader(file),1,None) if Path(file_path).exists() else [] #type: ignore

    with file:
        for timestamp in timestamps:
            window = []
            for row in rows:
                if float(row[0]) < timestamp-window_len: continue
                if float(row[0]) >= timestamp: break
                data = map(float,row[1:])
                window.append(next(data) if len(row) == 2 else tuple(data))
            yield window

def dems(pid, timestamps):
    df = pd.read_csv(f'{data_dir}/Baseline.csv')
    i = df["pid"].tolist().index(pid)
    return df.to_numpy()[[i]*len(timestamps), 1:].tolist()

def cacher(sensor,pid,ts,maker,*args):
    filename = f"{sensor}_{int(sum(ts))}_{args}_{pid}.pkl"
    features = load_feats(filename)
    if features: return features
    features = maker(pid,ts,*args)
    save_feats(filename,features)
    return features

def hrs(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/watch/{pid}/HeartRate.csv", timestamps, secs):
        w = list(filter(None,w))
        if w: features.append([np.mean(w),np.std(w)])
        else: features.append([float('nan')]*2)
    assert len(set(map(len,features))) == 1, 'hrs'
    return scale(features)

def scs1(pid, timestamps, secs):
    features = []
    if features: return features

    for w in wins(f"{data_dir}/watch/{pid}/StepCount.csv", timestamps, secs):
        if len(w)>1: features.append([np.mean(np.diff(w)),np.std(np.diff(w))])
        else: features.append([float('nan')]*2)
    assert len(set(map(len,features))) == 1, 'scs1'

    return scale(features)

def scs2(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/watch/{pid}/StepCount.csv", timestamps, secs):
        if len(w)>1: features.append([np.max(w)-np.min(w)])
        else: features.append([float('nan')])
    assert len(set(map(len,features))) == 1, 'scs2'
    return scale(features)

def lins1(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/LinearAcceleration.csv", timestamps, secs):
        if w: features.append([*np.var(w,axis=0),*np.percentile([np.linalg.norm(w,axis=1)],q=[10,50,90])])
        else: features.append([float('nan')]*6)
    assert len(set(map(len,features))) == 1, 'lins1'
    return scale(features)

def lins2(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/watch/{pid}/LinearAcceleration.csv", timestamps, secs):
        if w: features.append([*np.var(w,axis=0),*np.percentile([np.linalg.norm(w,axis=1)],q=[10,50,90])])
        else: features.append([float('nan')]*6)
    assert len(set(map(len,features))) == 1, 'lins2'
    return scale(features)

def lins3(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/LinearAcceleration.csv", timestamps, secs):
        if w: features.append([np.mean(np.linalg.norm(w,axis=1)), *np.std(w,axis=0)])
        else: features.append([float('nan')]*4)
    assert len(set(map(len,features))) == 1, 'lins3'
    return scale(features)

def lins4(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/watch/{pid}/LinearAcceleration.csv", timestamps, secs):
        if w: features.append([np.mean(np.linalg.norm(w,axis=1)), *np.std(w,axis=0)])
        else: features.append([float('nan')]*4)
    assert len(set(map(len,features))) == 1, 'lins2'
    return scale(features)

def bats1(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Battery.csv", timestamps, secs):
        w = [float(w)/100 for w in w]
        if len(w)==1: features.append([0,float('nan'),float('nan')])
        elif len(w)>1: features.append([np.max(w)-np.min(w),np.mean(np.diff(w)),np.std(np.diff(w))])
        else: features.append([float('nan')]*3)
        assert len(set(map(len,features))) == 1, 'bats1'
    return features

def bats2(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Battery.csv", timestamps, secs):
        w = [float(w)/100 for w in w]
        if w: features.append([np.mean(w),np.max(w)-np.min(w)])
        else: features.append([float('nan')]*2)
        assert len(set(map(len,features))) == 1, 'bats2'
    return features

def peds1(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Pedometer.csv", timestamps, secs):
        if len(w)==1: features.append([float('nan'),0,float('nan')])
        elif len(w)>1: features.append([np.mean(np.diff(w)),np.max(w)-np.min(w),np.std(np.diff(w))])
        else: features.append([float('nan')]*3)
        assert len(set(map(len,features))) == 1, 'peds1'
    return scale(features)

def peds2(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Pedometer.csv", timestamps, secs):
        if len(w)>1: features.append([np.max(w)-np.min(w)])
        else: features.append([float('nan')])
        assert len(set(map(len,features))) == 1, 'peds2'
    return scale(features)

def locs1(pid, timestamps, secs, freq, lmin, lmax):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Location.csv", timestamps, secs):
        if w: features.append([*np.mean(w,axis=0)[1:]])
        else: features.append([float('nan')]*2)
    out = TheoryGridCellSpatialRelationEncoder(frequency_num=freq,min_radius=lmin,max_radius=lmax,freq_init='geometric').make_input_embeds(features)
    return [out] if len(timestamps) == 1 else out

def locs2(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Location.csv", timestamps, secs):
        if w: features.append([*np.mean(w,axis=0)[1:]])
        else: features.append([float('nan')]*2)
    return features

def tims(timestamps,tzs):
    hour, day = 60*60, 60*60*24
    for timestamp,tz in zip(timestamps,tzs):
        if np.isnan(timestamp): 
            yield [float('nan')]*3
        else:
            if   tz == "-04:00": timestamp -= 4*hour
            elif tz == "-05:00": timestamp -= 5*hour
            time_of_day = (timestamp/day) % 1
            day_of_week = (int(timestamp/day)+4) % 7
            is_weekend = day_of_week in [0,6]
            is_weekday = day_of_week in [1,2,3,4,5]
            yield [time_of_day,int(is_weekend),int(is_weekday)]

def save_feats(filename,feats):
    if not Path(f"{data_dir}/feats/{filename}").exists():
        with open(f"{data_dir}/feats/{filename}", "wb") as f: # Use "wb" for binary write mode
            pickle.dump(feats, f)

def load_feats(filename):
    if not Path(f"{data_dir}/feats/{filename}").exists():
        return None
    else:
        try:
            with open(f"{data_dir}/feats/{filename}", "rb") as f: # Use "wb" for binary write mode
                return pickle.load(f)
        except:
            Path(f"{data_dir}/feats/{filename}").unlink()
            return None

def make_xyg1(work_item):
    (pid,ts,tz,ys,args,secs) = work_item

    fs = []

    if args[0]: fs.append(list(tims(ts,tz)))
    if args[1]: fs.append(list(cacher("hrs"  ,pid,ts,hrs  ,args[1])))
    if args[2]: fs.append(list(cacher("scs1" ,pid,ts,scs1 ,args[2])))
    if args[3]: fs.append(list(cacher("lins1",pid,ts,lins1,args[3])))
    if args[4]: fs.append(list(cacher("lins2",pid,ts,lins2,args[4])))
    if args[5]: fs.append(list(cacher("bats1",pid,ts,bats1,args[5])))
    if args[6]: fs.append(list(cacher("peds1",pid,ts,peds1,args[6])))
    if args[7]: fs.append(list(cacher("locs1",pid,ts,locs1,*args[7])))
    if args[8]: fs.append(list(cacher("locs2",pid,ts,locs2,args[8])))

    if args[10]:
        for f in fs: add1(f)

    if args[9]: fs.append(list(dems(pid,ts)))

    os = list(ys)
    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    gs = [pid]*len(ys)

    for sec in (secs or []):
        nts = [t-sec for t in ts]
        nxs,nys,ngs = make_xyg2((pid,nts,tz,os,args,0))
        xs += nxs
        ys += nys
        gs += ngs

    return xs,ys,gs

def make_xyg2(work_item):
    (pid,ts,tz,ys,args,secs) = work_item

    fs = []

    if args[0]: fs.append(list(tims(ts,tz)))
    if args[1]: fs.append(list(cacher("hrs"  ,pid,ts,hrs  ,args[1])))
    if args[2]: fs.append(list(cacher("scs2" ,pid,ts,scs2 ,args[2])))
    if args[3]: fs.append(list(cacher("lins3",pid,ts,lins3,args[3])))
    if args[4]: fs.append(list(cacher("lins4",pid,ts,lins4,args[4])))
    if args[5]: fs.append(list(cacher("bats2",pid,ts,bats2,args[5])))
    if args[6]: fs.append(list(cacher("peds2",pid,ts,peds2,args[6])))
    if args[7]: fs.append(list(cacher("locs1",pid,ts,locs1,*args[7])))
    if args[8]: fs.append(list(cacher("locs2",pid,ts,locs2,args[8])))

    if args[10]:
        for f in fs: add1(f)

    if args[9]: fs.append(list(dems(pid,ts)))

    os = list(ys)
    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    gs = [pid]*len(ys)

    for sec in (secs or []):
        if sec != 0:
            nts = [t-sec for t in ts]
            nxs,nys,ngs = make_xyg2((pid,nts,tz,os,args,0))
            xs += nxs
            ys += nys
            gs += ngs

    return xs,ys,gs

can_predict = emotions_df.copy().sort_values(["ParticipantId","DeliveredTimestampUtc"])

def work_items(tims:bool,hrs:int,scs:int,lins1:int,lins2:int,bats:int,peds:int,locs1,locs2:int,dems:bool,add1:bool,event:str,secs=[]):

    df = can_predict[~can_predict["SubmissionTimestampUtc"].isna()]

    for pid in sorted(df["ParticipantId"].drop_duplicates().tolist()):
        ptc  = df[df.ParticipantId == pid]
        tss  = ptc["SubmissionTimestampUtc" if event == "sub" else "DeliveredTimestampUtc"].tolist() 
        tzs  = ptc["LocalTimeZone"].tolist()

        y0s = torch.tensor(is_gt(ptc["ER Interest (z)"],0).tolist())
        y1s = torch.tensor(is_lt(ptc['Response Time (min)'],10).tolist())
        y2s = torch.tensor(is_gt(ptc["State Anxiety (z)"],0).tolist())
        y3s = torch.tensor(is_gt(ptc["State Anxiety"], 1).tolist())
        y4s = torch.tensor(ptc["Response Time (log sec)"].tolist())
        y5s = torch.tensor(ptc["ER Interest (z)"].tolist())
        y6s = torch.tensor(ptc["State Anxiety (z)"].tolist())

        ys = torch.hstack((
            y0s.unsqueeze(1),
            y1s.unsqueeze(1),
            y2s.unsqueeze(1),
            y3s.unsqueeze(1),
            y4s.unsqueeze(1),
            y5s.unsqueeze(1),
            y6s.unsqueeze(1)
        )).tolist()

        args = [tims,hrs,scs,lins1,lins2,bats,peds,locs1,locs2,dems,add1]

        yield pid,tss,tzs,ys,args,secs

#with ProcessPoolExecutor(max_workers=20) as executor:
X,Y,G = zip(*map(make_xyg2, work_items(True,0,0,300,0,300,300,[300,2,1,2],0,False,True,"del")))

Y = torch.tensor(list(chain.from_iterable(Y))).float()
G = torch.tensor(list(chain.from_iterable(G))).int()
G = G[~torch.isnan(Y[:,[0,1]]).any(dim=1)]

testable_G = cb.CobaRandom(1).shuffle([k for k,n in Counter(G.tolist()).items() if n >= 30])

#########################################################################################

class FeedForward(torch.nn.Sequential):
    """A Generic implementation of Feedforward Neural Network"""

    class SkipModule(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers
        def forward(self,X):
            return X + self.layers(X)
        
    class ForceOneModule(torch.nn.Module):
        def forward(self,X):
            return torch.ones(size=(X.shape[0],1)).float()

    def make_layer(self,curr_dim,spec):
        if isinstance(spec,float):
            return torch.nn.Dropout(spec), curr_dim
        if curr_dim is None and isinstance(spec,int):
            return None, spec
        if isinstance(spec,int):
            return torch.nn.Linear(curr_dim,spec),spec
        if spec == 'r':
            return torch.nn.ReLU(),curr_dim
        if spec == 'l':
            return torch.nn.LayerNorm(curr_dim),curr_dim
        if spec == 'b':
            return torch.nn.BatchNorm1d(curr_dim), curr_dim
        if spec == 's':
            return torch.nn.Sigmoid(),curr_dim
        if spec == '1':
            return FeedForward.ForceOneModule(), 1
        if isinstance(spec,list):                
            return FeedForward.SkipModule(FeedForward([curr_dim] + spec)), curr_dim
        raise Exception(f"Bad Layer: {spec}")

    def __init__(self, specs, rng=1):
        """Instantiate a Feedfoward network according to specifications.

        Args:
            specs: A sequence of layer specifications as follows:
                -1 -- replaced with the input feature width
                <int> -- a LinearLayer with output width equal to <int>
                [0,1] -- a Dropout layer with the given probability
                'l' -- a LayerNorm
                'b' -- a BatchNorm1d
                'r' -- a ReLU layer
                's' -- a Sigmoid layer
                [] -- a skip layer with the given specifications
        """

        torch.manual_seed(rng)
        layers,width = [],None
        for spec in specs:
            layer,width = self.make_layer(width,spec)
            if layer: layers.append(layer)
        super().__init__(*(layers or [torch.nn.Identity()]))
        self.params = {"specs": specs, "rng": rng }

class MyEnvironment:
    def __init__(self, a, L, g, rng, v=2):
        self.params = {'rng': rng, 'trn':a, 'l':L, 'v':v, 'g':g }
        self.X = None
        self.Y = None
        self.G = None
        self.a = list(a)
        self.L = L
        self.g = g
        self.v = v
        self.a[7] = [a[7],2,1,10] if a[7] else None

    def get_data(self):
        import torch
        import itertools as it

        if self.X is not None: return self.X,self.Y,self.G
        make = make_xyg1 if self.v == 1 else make_xyg2

        X,Y,G = zip(*map(make, work_items(*self.a)))

        X = torch.tensor(list(it.chain.from_iterable(X))).float()
        Y = torch.tensor(list(it.chain.from_iterable(Y))).float()
        G = torch.tensor(list(it.chain.from_iterable(G))).int()

        self.X,self.Y,self.G = X,Y,G

        if X.shape[0] == 0: return

        any_na = torch.isnan(Y[:,[0,1]]).any(dim=1)
        X = X[~any_na]
        Y = Y[~any_na].float()
        G = G[~any_na]

        rng_indexes = cb.CobaRandom(self.params['rng']).shuffle(range(len(X)))

        self.X,self.Y,self.G = X[rng_indexes],Y[rng_indexes],G[rng_indexes]

        return self.X,self.Y,self.G

class MyEvaluator:
    def __init__(self, s1, s2, s3, dae_steps, dae_dropn, ws_steps0, ws_drop0, ws_steps1, pers_lrn_cnt, pers_mem_cnt, pers_mem_rpt, pers_mem_rcl, pers_rank, n_models, weighted):

        self.s1  = s1  #dae sep-sl
        self.s2  = s2  #sep-sl
        self.s3  = s3  #one-sl pers

        self.dae_steps = dae_steps
        self.dae_dropn = dae_dropn

        self.ws_steps0 = ws_steps0
        self.ws_drop0  = ws_drop0
        self.ws_steps1 = ws_steps1

        self.pers_lrn_cnt = pers_lrn_cnt
        self.pers_mem_cnt = pers_mem_cnt
        self.pers_mem_rpt = pers_mem_rpt
        self.pers_mem_rcl = pers_mem_rcl
        self.pers_rank    = pers_rank

        self.n_models = n_models
        self.weighted = weighted

        self.params = { 's1': s1, 's2':s2, 's3':s3, 'dae': (dae_steps,dae_dropn), 'ws': (ws_steps0,ws_drop0,ws_steps1), 'pers': (pers_lrn_cnt,pers_mem_cnt,pers_mem_rpt,pers_mem_rcl,pers_rank), 'n_models': n_models, 'weighted': weighted }

    def evaluate(self, env, lrn):
        from copy import deepcopy
        from numpy import nanmean
        from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_score, recall_score
        from collections import Counter

        X,Y,G = env.get_data()
        if len(X) == 0: return

        torch.set_num_threads(1)

        def make_weights(G):
            W = torch.zeros((len(G),1))
            weights = Counter(G.tolist())
            for g,w in weights.items():
                W[G==g] = 1/w
            return (W / W.max())
        
        def get_trn_tst(G,g):
            is_tst = sum(G == i for i in g).bool() #type: ignore
            return ~is_tst, is_tst

        def get_scores(X,Y,G,g,N,env):

            torch.manual_seed(env.params['rng'])

            is_trn, is_tst = get_trn_tst(G,g)
            trn_X, trn_Y, trn_G = X[is_trn], Y[is_trn], G[is_trn]
            tst_X, tst_Y, tst_G = X[is_tst], Y[is_tst], G[is_tst]
            #469 [1] -- 492 [1] -- 472 [1] -- 432 [1] -- 431 [1] -- 430 [1] -- 413 [1]
            #421 [0] -- 498 [1]
            try:
                next(StratifiedShuffleSplit(1,train_size=N).split(tst_X,tst_Y))
            except Exception as ex:
                return
            
            if len(set(tst_Y.squeeze().tolist())) == 1:
                return
            
            return

            n_feats = X.shape[1]
            n_persons = len(set(trn_G.tolist()))
            n_tasks = Y.shape[1]

            _s1 = [n_feats if f == 'x' else f if f != '-x' else n_feats*n_persons for f in self.s1]
            _s2 = [n_feats if f == 'x' else f                                     for f in self.s2]
            _s3 = [n_feats if f == 'x' else f                                     for f in self.s3]

            _s1 = [n_tasks if f == 'y' else f for f in _s1]
            _s2 = [n_tasks if f == 'y' else f for f in _s2]
            _s3 = [n_tasks if f == 'y' else f for f in _s3]

            if _s2 and _s2[-1] == -1: _s2 = (*(_s2)[:-1], n_persons*n_tasks)
            if _s3 and _s3[ 0] == -1: _s3 = (n_persons*n_tasks, *(_s3)[1:])

            mods_opts = []
            opts = []

            for _ in range(self.n_models):
                s1 = FeedForward(_s1)
                s2 = FeedForward(_s2)
                s3 = FeedForward(_s3)

                s1_children = list(s1.children())
                s2_children = list(s2.children())

                sa = torch.nn.Sequential(*s1_children[len(s1_children)-self.dae_dropn:])
                s1 = torch.nn.Sequential(*s1_children[:len(s1_children)-self.dae_dropn])

                sb = torch.nn.Sequential(*s2_children[len(s2_children)-self.ws_drop0:])
                s2 = torch.nn.Sequential(*s2_children[:len(s2_children)-self.ws_drop0])

                s1opt = COCOB(s1.parameters()) if list(s1.parameters()) else None
                saopt = COCOB(sa.parameters()) if list(sa.parameters()) else None
                s2opt = COCOB(s2.parameters()) if list(s2.parameters()) else None
                sbopt = COCOB(sb.parameters()) if list(sb.parameters()) else None
                s3opt = COCOB(s3.parameters()) if list(s3.parameters()) else None

                mods = [s1,sa,s2,sb,s3]
                opts = [s1opt,saopt,s2opt,sbopt,s3opt]
                mods_opts.append([mods,opts])

            for mods,_ in mods_opts:
                for m in mods: m.train()

            for mods,opts in mods_opts:
                [s1,sa,s2,sb,s3] = mods
                [s1opt,saopt,s2opt,sbopt,s3opt] = opts

                if _s1 and self.dae_steps:
                    opts = list(filter(None,[s1opt,saopt]))
                    X,G,W = trn_X,trn_G,make_weights(trn_G)

                    if _s1[-1] != n_feats*n_persons:
                        Z = X
                    else:
                        i = defaultdict(lambda c= count(0):next(c))
                        I = torch.tensor([[i[g]] for g in G.tolist()]) * n_feats + torch.arange(n_feats).unsqueeze(0)
                        R = torch.arange(len(X)).unsqueeze(1)
                        Z = torch.full((len(X),len(i)*n_feats), float('nan'))
                        Z[R,I] = X

                    torch_dataset = torch.utils.data.TensorDataset(X,Z,W)
                    torch_loader  = torch.utils.data.DataLoader(torch_dataset,batch_size=8,drop_last=True,shuffle=True)

                    loss = torch.nn.L1Loss()
                    for _ in range(self.dae_steps):
                        for (_X,_z,_w) in torch_loader:
                            for o in opts: o.zero_grad()
                            loss(sa(s1(_X.nan_to_num()))[~_z.isnan()],_z[~_z.isnan()]).backward()                        
                            for o in opts: o.step()

                if self.ws_steps0:
                    opts = list(filter(None,[s1opt,s2opt,sbopt]))
                    for o in opts: o.zero_grad()

                    X, Y, G, W = trn_X,trn_Y,trn_G,make_weights(trn_G)

                    if _s2[-1] != n_tasks*n_persons:
                        Z = Y
                    else:
                        i = defaultdict(lambda c= count(0):next(c))
                        I = torch.tensor([[i[g]] for g in G.tolist()]) * n_tasks + torch.arange(n_tasks).unsqueeze(0)
                        R = torch.arange(len(Y)).unsqueeze(1)
                        Z = torch.full((len(G),len(i)*n_tasks), float('nan'))
                        Z[R,I] = Y

                    torch_dataset = torch.utils.data.TensorDataset(X,Z,W)
                    torch_loader  = torch.utils.data.DataLoader(torch_dataset,batch_size=8,drop_last=True,shuffle=True)

                    for _ in range(self.ws_steps0):
                        for _X,_z,_w in torch_loader:
                            for o in opts: o.zero_grad()                        
                            loss = torch.nn.BCEWithLogitsLoss(weight=_w.squeeze() if 2 in self.weighted else None)
                            loss(sb(s2(s1(_X.nan_to_num())))[~_z.isnan()],_z[~_z.isnan()]).backward()
                            for o in opts: o.step()

                if self.ws_steps1:
                    opts = list(filter(None,[s3opt] if self.ws_steps0 else [s1opt,s2opt,s3opt]))
                    for o in opts: o.zero_grad()

                    X, Y, G, W = trn_X,trn_Y,trn_G,make_weights(trn_G)

                    torch_dataset = torch.utils.data.TensorDataset(X,Y,W)
                    torch_loader  = torch.utils.data.DataLoader(torch_dataset,batch_size=8,drop_last=True,shuffle=True)

                    loss = torch.nn.BCEWithLogitsLoss()
                    for _ in range(self.ws_steps1):
                        for _X,_y,_w in torch_loader:
                            for o in opts: o.zero_grad()
                            loss = torch.nn.BCEWithLogitsLoss(weight=_w if 3 in self.weighted else None)
                            loss(s3(s2(s1(_X.nan_to_num()))),_y).backward()
                            for o in opts: o.step()

            for mods,_ in mods_opts:
                for m in mods: m.eval()

            def predict(X):
                preds = torch.tensor(0)
                for mods,_ in mods_opts:
                    [s1,_,s2,_,s3] = mods
                    if s3: preds = preds + torch.sigmoid(s3(s2(s1(X.nan_to_num()))))
                return preds/len(mods_opts)

            def score(X,Y):
                out = dict()
                with torch.no_grad():
                    probs = predict(X)
                    preds = (probs>=.5).float()
                    for i in range(n_tasks):

                        one = len(set(Y[:,i].tolist())) == 1

                        tp = ((preds[:,i]==1) & (Y[:,i]==1)).float().mean().item()
                        tn = ((preds[:,i]==0) & (Y[:,i]==0)).float().mean().item()
                        fp = ((preds[:,i]==1) & (Y[:,i]==0)).float().mean().item()
                        fn = ((preds[:,i]==0) & (Y[:,i]==1)).float().mean().item()

                        out[f"auc{i}"] = float('nan') if one else roc_auc_score(Y[:,i],probs[:,i])
                        out[f"bal{i}"] = float('nan') if one else balanced_accuracy_score(Y[:,i],preds[:,i])
                        out[f"sen{i}"] = float('nan') if one else tp/(tp+fn)
                        out[f"spe{i}"] = float('nan') if one else tn/(tn+fp)

                        for j in [0,1]:
                            out[f"f1{j}{i}" ] = float('nan') if one else f1_score(Y[:,i],preds[:,i],pos_label=j)
                            out[f"pre{j}{i}"] = float('nan') if one else precision_score(Y[:,i],preds[:,i],pos_label=j,zero_division=0)
                            out[f"rec{j}{i}"] = float('nan') if one else recall_score(Y[:,i],preds[:,i],pos_label=j)

                        out[f"f1m{i}"] = float('nan') if one else f1_score(Y[:,i],preds[:,i],average='macro')
                        out[f"f1w{i}"] = float('nan') if one else f1_score(Y[:,i],preds[:,i],average='weighted')

                return out

            scores = [ [] for _ in range(N+1) ]
            unchanged_mods_opts = mods_opts

            X,Y = tst_X, tst_Y

            for j in range(15):

                mods_opts = []
                for mods,opts in unchanged_mods_opts:
                    mods = deepcopy(mods)
                    opts = deepcopy(opts)
                    mods_opts.append([mods,opts])

                trn,tst = next(StratifiedShuffleSplit(1,train_size=N,random_state=j).split(X,Y))
                trn_X,trn_Y = X[trn],Y[trn]
                scr_X,scr_Y = X[tst],Y[tst]

                for mods,opts in mods_opts:
                    if not self.pers_rank:
                        opts[-1] = COCOB(mods[-1].parameters()) if opts[-1] else None

                lrnxs = [[] for _ in range(len(mods_opts))]
                lrnys = [[] for _ in range(len(mods_opts))]
                memss = [[] for _ in range(len(mods_opts))]

                scores[0].append(score(scr_X, scr_Y))

                rng = cb.CobaRandom(1)

                loss = torch.nn.BCEWithLogitsLoss()
                for i in range(N):

                    for lrnx,lrny,mems,(mods,opts) in zip(lrnxs,lrnys,memss,mods_opts):
                        [s1,_,s2,_,s3 ] = mods
                        [_,_,_,_,s3opt] = opts

                        x,y = trn_X[i,:], trn_Y[i,:]

                        if self.pers_lrn_cnt:
                            lrnx.append(x)
                            lrny.append(y)

                            if self.pers_mem_cnt: 
                                mems.append([x,y,self.pers_mem_rpt])

                            if len(mems) > self.pers_mem_cnt:
                                rng.shuffle(mems, inplace=True)
                                for j in reversed(range(self.pers_mem_rcl)):
                                    if j >= len(mems): continue
                                    x,y,n = mems[j]
                                    lrnx.append(x)
                                    lrny.append(y)
                                    if n == 1: mems.pop(j)
                                    else: mems[j] = [x,y,n-1]

                            if len(lrnx) >= self.pers_lrn_cnt:
                                x = torch.stack(lrnx[:self.pers_lrn_cnt])
                                y = torch.stack(lrny[:self.pers_lrn_cnt])

                                if s3opt:
                                    if s3opt: s3opt.zero_grad()
                                    loss(s3(s2(s1(x.nan_to_num()))),y).backward()
                                    if s3opt: s3opt.step()

                                del lrnx[:self.pers_lrn_cnt]
                                del lrny[:self.pers_lrn_cnt]

                    scores[i+1].append(score(scr_X, scr_Y))

            for s in scores:
                yield {k:nanmean([s_[k] for s_ in s]) for k in s[0].keys()}

        yield from get_scores(X,Y[:,env.L],G,env.g,20,env)

testable_G = cb.CobaRandom(1).shuffle([k for k,n in Counter(G.tolist()).items() if n >= 30])

x = [0,300]
L = [[0],[1]]
A = product([True],x,x,x,x,x,x,x,[0],[True,False],[True],["del"])

envs = [ MyEnvironment(a,l,[g],rng) for a in A for g in testable_G for rng in range(1) for l in L ]

lrns = [ None ]
vals = [
    MyEvaluator((), ('x',120,'l','r',90,'l','r',-1), (90,'y'), 0, 0, 4, 1, 4, 3, 2, 2, 2, 0, 1, []),
]

cb.Experiment(envs,lrns,vals).run(processes=1,quiet=True) #type: ignore