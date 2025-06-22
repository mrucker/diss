import csv
from pathlib import Path
from collections import defaultdict, Counter
from itertools import islice, chain, count, product, repeat
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
from sklearn.model_selection import StratifiedShuffleSplit
from parameterfree import COCOB
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from IPython.display import clear_output

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

    def scale(df, group_col, scale_col, scaler, postfix):
        for gid in set(df[group_col]):
            df.loc[df[group_col]==gid,[scale_col+postfix]] = scaler.fit_transform(df.loc[df[group_col]==gid,[scale_col]])

    def drop_all1_ends(df):

        drop = df.copy()
        keep = df.copy()
        
        drop = drop[drop["State Anxiety"]!= 1]
        drop = drop.groupby("ParticipantId")["RelDay"].max().reset_index()
        drop = drop.rename(columns={"RelDay":"Last Day With Anxiety > 1"})
        drop = drop[drop["Last Day With Anxiety > 1"] <= 8 ]

        for pid,day in drop.itertuples(index=False):
            is_not_pid = keep["ParticipantId"] != pid
            is_lt_day  = keep["RelDay"] < day
            keep = keep[is_not_pid | is_lt_day]

        return keep

    def get_trimmed_pids(df):
        df = df.copy()
        df = df[~df["ER Interest"].isna()]
        df = df.groupby(["ParticipantId","RelDay"]).size().reset_index(drop=False)
        df = df.groupby(['ParticipantId']).size().reset_index(drop=False).rename(columns={0:"Count"})
        return df.loc[df.Count < 6,'ParticipantId'].tolist()

    runs_df = pd.read_csv(f'{data_dir}/Runs.csv')
    states_df = pd.read_csv(f'{data_dir}/States.csv')
    emotions_df = pd.read_csv(f'{data_dir}/Emotions.csv')
    participant_df = pd.read_csv(f'{data_dir}/Participants.csv')

    emotions_df = emotions_df[emotions_df["WatchDataQuality"] == "Good"]

    emotions_df["ER Interest"] = pd.to_numeric(emotions_df["ER Interest"], errors='coerce')
    emotions_df["Phone ER Interest"] = pd.to_numeric(emotions_df["Phone ER Interest"], errors='coerce')
    emotions_df["Response Time (min)"] = (emotions_df["SubmissionTimestampUtc"] - emotions_df["DeliveredTimestampUtc"])/60

    runs_df = add_day_columns(runs_df, "DeliveredTimestampUtc", participant_df)
    states_df = add_day_columns(states_df, "TimestampUtc", participant_df)
    emotions_df = add_day_columns(emotions_df, "OpenedTimestampUtc", participant_df)

    to_trim = get_trimmed_pids(emotions_df)

    runs_df = runs_df[~runs_df["ParticipantId"].isin(to_trim)]
    states_df = states_df[~states_df["ParticipantId"].isin(to_trim)]
    emotions_df = emotions_df[~emotions_df["ParticipantId"].isin(to_trim)]

    runs_df = runs_df[runs_df["RelDay"] < 11]
    states_df = states_df[states_df["RelDay"] < 11]
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

def wins(file_path, timestamps, window_len):
    file = open(file_path) if Path(file_path).exists() else nullcontext()
    rows = islice(csv.reader(file),1,None) if Path(file_path).exists() else [] #type: ignore

    with file:
        for timestamp in timestamps:
            window = []
            for row in rows:
                if float(row[0]) < timestamp-window_len: continue
                if float(row[0]) > timestamp: break
                data = map(float,row[1:])
                window.append(next(data) if len(row) == 2 else tuple(data))
            yield window

def dems(pid, timestamps):
    df = pd.read_csv(f'{data_dir}/Baseline.csv',header=None)
    i = df[0].tolist().index(pid)
    return df.to_numpy()[:, 2:-109][[i]*len(timestamps),:]

def add1(X):
    for x,z in zip(X,np.isnan(X).all(axis=1).astype(int)):
        x.append(z)
    return X

def hrs(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/HeartRate.csv", timestamps, secs):
        w = list(filter(None,w))
        if w: features.append([np.mean(w),np.std(w)])
        else: features.append([float('nan')]*2)
    assert len(set(map(len,features))) == 1, 'hrs'
    return StandardScaler().fit_transform(features).tolist() #type: ignore

def scs(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/watch/{pid}/StepCount.csv", timestamps, secs):
        if w: features.append([np.mean(np.diff(w)),np.std(np.diff(w))])
        else: features.append([float('nan')]*2)
    assert len(set(map(len,features))) == 1, 'scs'
    return StandardScaler().fit_transform(features).tolist() #type: ignore

def lins1(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/LinearAcceleration.csv", timestamps, secs):
        if w: features.append([*np.var(w,axis=0),*np.percentile([np.linalg.norm(w,axis=1)],q=[10,50,90])])
        else: features.append([float('nan')]*6)
    assert len(set(map(len,features))) == 1, 'lins1'
    return StandardScaler().fit_transform(features).tolist() #type: ignore

def lins2(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/watch/{pid}/LinearAcceleration.csv", timestamps, secs):
        if w: features.append([*np.var(w,axis=0),*np.percentile([np.linalg.norm(w,axis=1)],q=[10,50,90])])
        else: features.append([float('nan')]*6)
    assert len(set(map(len,features))) == 1, 'lins2'
    return StandardScaler().fit_transform(features).tolist() #type: ignore

def bats(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Battery.csv", timestamps, secs):
        w = [float(w)/100 for w in w]
        if w: features.append([np.max(w)-np.min(w),np.mean(np.diff(w)),np.std(np.diff(w))])
        else: features.append([float('nan')]*3)
        assert len(set(map(len,features))) == 1, 'bats'
    return features

def peds(pid, timestamps, secs):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Pedometer.csv", timestamps, secs):
        if w: features.append([np.mean(np.diff(w)),np.max(w)-np.min(w),np.std(np.diff(w))])
        else: features.append([float('nan')]*3)
        assert len(set(map(len,features))) == 1, 'peds'
    return StandardScaler().fit_transform(features).tolist() #type: ignore

def locs1(pid, timestamps, secs, init, freq, lmin, lmax):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Location.csv", timestamps, secs):
        if w: features.append([*np.mean(w,axis=0)[1:]])
        else: features.append([float('nan')]*2)
    out = TheoryGridCellSpatialRelationEncoder(frequency_num=freq,max_radius=lmax,min_radius=lmin,freq_init=init).make_input_embeds(features)
    return [out] if len(timestamps) == 1 else out

def locs2(pid, timestamps, secs, freq, lmax, lmin, init):
    features = []
    for w in wins(f"{data_dir}/phone/{pid}/Location.csv", timestamps, secs):
        if w: features.append([*np.mean(w,axis=0)[1:]])
        else: features.append([float('nan')]*2)
    return features

def tims(timestamps,tzs):
    hour, day = 60*60, 60*60*24
    for timestamp,tz in zip(timestamps,tzs):
        if   tz == "-04:00": timestamp -= 4*hour
        elif tz == "-05:00": timestamp -= 5*hour
        time_of_day = (timestamp/day) % 1
        day_of_week = (int(timestamp/day)+4) % 7
        is_weekend = day_of_week in [0,6]
        is_weekday = day_of_week in [1,2,3,4,5]
        yield [time_of_day,int(is_weekend),int(is_weekday)]

def make_xyg1(work_item):
    (pid,ts,tz,ys,args) = work_item
    #hrs, scs, lins, bats, peds, locs
    fs = [
        locs1(pid,ts,*args[5]),
        lins1(pid,ts,*args[2]),
        tims(ts,tz),
        bats(pid,ts,*args[3]),
        peds(pid,ts,*args[4])
    ]
    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    ys = [float(y<np.mean(ys)) for y in ys]
    gs = [pid]*len(ys)
    return xs,ys,gs

def make_xyg2(work_item):
    (pid,ts,tz,ys,args) = work_item
    #hrs, scs, lins, bats, peds, locs
    fs = [
        locs1(pid,ts,*args[5]),
        lins1(pid,ts,*args[2]),
        lins2(pid,ts,*args[2]),
        scs(pid,ts,*args[1]),
        hrs(pid,ts,*args[0]),
        tims(ts,tz),
        bats(pid,ts,*args[3]),
        peds(pid,ts,*args[4])
    ]
    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    ys = [float(y<np.mean(ys)) for y in ys]
    gs = [pid]*len(ys)
    return xs,ys,gs

def make_xyg3(work_item):
    (pid,ts,tz,ys,args) = work_item
    #hrs, scs, lins, bats, peds, locs
    fs = [
        locs1(pid,ts,*args[5]),
        lins1(pid,ts,*args[2]),
        scs(pid,ts,*args[1]),
        hrs(pid,ts,*args[0]),
        tims(ts,tz),
        bats(pid,ts,*args[3]),
        peds(pid,ts,*args[4])
    ]
    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    ys = [float(y<np.mean(ys)) for y in ys]
    gs = [pid]*len(ys)
    return xs,ys,gs

def make_xyg4(work_item):
    (pid,ts,tz,ys,args) = work_item
    #hrs, scs, lins, bats, peds, locs
    fs = [
        locs1(pid,ts,*args[5]),
        lins1(pid,ts,*args[2]),
        tims(ts,tz),
        bats(pid,ts,*args[3]),
        peds(pid,ts,*args[4]),
        dems(pid,ts)
    ]
    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    ys = [float(y<np.mean(ys)) for y in ys]
    gs = [pid]*len(ys)
    return xs,ys,gs

def make_xyg5(work_item):
    (pid,ts,tz,ys,args,ind) = work_item
    #hrs, scs, lins, bats, peds, locs
    fs = [
        locs1(pid,ts,*args[5]),
        lins1(pid,ts,*args[2]),
        tims(ts,tz),
        bats(pid,ts,*args[3]),
        peds(pid,ts,*args[4])
    ]

    if ind:
        for f in fs:
            add1(f)

    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    ys = [float(y<np.mean(ys)) for y in ys]
    gs = [pid]*len(ys)
    return xs,ys,gs

can_predict = emotions_df[(emotions_df["WatchDataQuality"] == "Good") & ~emotions_df["ER Interest"].isna()].copy()
can_predict = can_predict.sort_values(["ParticipantId","SubmissionTimestampUtc"])

########################################################################################

class FeedForward(torch.nn.Sequential):
    """A Generic implementation of Feedforward Neural Network"""

    class SkipModule(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers
        def forward(self,X):
            return X + self.layers(X)

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
    def __init__(self, train_X, train_Y, train_G, test_X, test_Y, trn, g, rng):
        self.params = {'pid': g, 'rng': rng, 'trn':trn}
        self.train_X = train_X
        self.train_Y = train_Y.float()
        self.train_G = train_G
        self.test_X = test_X
        self.test_Y = test_Y.float()

    def ssl(self,neg,sr,yi):
        from itertools import compress, repeat, chain
        from operator import eq

        rng = cb.CobaRandom(self.params['rng'])
        rng_order = rng.shuffle(range(len(self.train_X)))

        X = self.train_X.tolist()
        Y = self.train_Y[:,yi]
        Y = list(map(tuple,Y.tolist()))

        X = list(map(X.__getitem__,rng_order))
        Y = list(map(Y.__getitem__,rng_order))

        eq_class  = {y: list(compress(X,map(eq,Y,repeat(y)))) for y in set(Y)}
        ne_class  = {y: list(chain(*[v for k,v in eq_class.items() if k != y ])) for y in set(Y)}

        def choose_unique(items,given_i):
            if len(items) == 1:  return items[0]
            for i in rng.randints(None,0,len(items)-1):
                if i != given_i:
                    return items[i]

        def choose_n(items,n):
            add_to_index = (indexes := set()).add if len(items) > n else (indexes := []).append
            for i in rng.randints(None,0,len(items)-1):
                add_to_index(i)
                if len(indexes)==n:
                    return [items[i] for i in indexes]

        if sr < 1:
            anchor, positive, negative = [], [], []

            for i in range(int(len(X)*sr)):
                x,y = X[i],Y[i]
                anchor.append(x)
                positive.append(choose_unique(eq_class[y],i))
                negative.append(choose_n     (ne_class[y],neg))
            yield torch.tensor(anchor).float(), torch.tensor(positive).float(), torch.tensor(negative).float()

        else:
            for _ in range(sr):
                anchor, positive, negative = [], [], []
                for i in range(len(X)):
                    x,y = X[i],Y[i]
                    anchor.append(x)
                    positive.append(choose_unique(eq_class[y],i))
                    negative.append(choose_n     (ne_class[y],neg))

                yield torch.tensor(anchor).float(), torch.tensor(positive).float(), torch.tensor(negative).float()

    def train(self):
        new_id = defaultdict(lambda c= count(0):next(c))
        sampler = RandomOverSampler(random_state=self.params['rng'])
        _Z = [ new_id[z] for z in zip(self.train_Y.squeeze().tolist(),self.train_G.squeeze().tolist()) ]
        _I = sampler.fit_resample(torch.arange(len(self.train_X)).unsqueeze(1),_Z)[0] #type: ignore
        _Y = self.train_Y[_I.squeeze()]
        _X = self.train_X[_I.squeeze()]
        _G = self.train_G[_I.squeeze()]
        rng_indexes = cb.CobaRandom(self.params['rng']).shuffle(range(len(_X)))
        return _X[rng_indexes,:], _Y[rng_indexes,:], _G[rng_indexes]
    
    def test(self):
        rng_indexes = cb.CobaRandom(self.params['rng']).shuffle(range(len(self.test_X)))
        return self.test_X[rng_indexes,:], self.test_Y[rng_indexes]

class MyEvaluator:
    def __init__(self, s1, s2, s3, dae_steps, dae_dropn, ws_steps0, ws_drop0, ws_steps1, pers_lrn_cnt, pers_mem_cnt, pers_mem_rpt, pers_mem_rcl, pers_rank, y, n_models, weighted):

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

        self.y = y
        self.n_models = n_models
        self.weighted = weighted

        self.params = { 's1': s1, 's2':s2, 's3':s3, 'dae': (dae_steps,dae_dropn), 'ws': (ws_steps0,ws_drop0,ws_steps1), 'pers': (pers_lrn_cnt,pers_mem_cnt,pers_mem_rpt,pers_mem_rcl,pers_rank), 'y': y, 'n_models': n_models, 'weighted': weighted }

    def evaluate(self, env, lrn):
        from statistics import mean
        from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_score, recall_score
        from copy import deepcopy
        from collections import Counter
        import peft
        
        rng = cb.CobaRandom(env.params['rng'])
        torch.manual_seed(env.params['rng'])
        torch.set_num_threads(1)

        def make_weighted(G):
            W = torch.zeros((len(G),1))
            weights = Counter(G.tolist())
            for g,w in weights.items():
                W[G==g] = 1/w
            return (W / W.max())

        mods_opts = []
        opts = []

        n_feats = env.test()[0].shape[1]
        n_persons = len(set(env.train()[2].tolist()))
        n_y = len(self.y)

        self.s1 = [n_feats if f == 'x' else f if f != '-x' else n_feats*n_persons for f in self.s1]
        self.s2 = [n_feats if f == 'x' else f for f in self.s2]
        self.s3 = [n_feats if f == 'x' else f for f in self.s3]

        if self.s2 and self.s2[-1] == -1: self.s2 = (*(self.s2)[:-1], n_persons*n_y)
        if self.s3 and self.s3[ 0] == -1: self.s3 = (n_persons*n_y, *(self.s3)[1:])

        for _ in range(self.n_models):
            s1 = FeedForward(self.s1)
            s2 = FeedForward(self.s2)
            s3 = FeedForward(self.s3)

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
            for l in mods: l.train()

        for mods,opts in mods_opts:
            [s1,sa,s2,sb,s3] = mods
            [s1opt,saopt,s2opt,sbopt,s3opt] = opts

            if self.s1 and self.dae_steps:
                opts = list(filter(None,[s1opt,saopt]))
                X,_,G = env.train()
                W = make_weighted(G)

                if self.s1[-1] != n_feats*n_persons:
                    Z = X
                else:
                    i = defaultdict(lambda c= count(0):next(c))
                    I = torch.tensor([[i[g]] for g in G.tolist()])*n_feats + torch.arange(n_feats).unsqueeze(0)
                    R = torch.arange(len(X)).unsqueeze(1)
                    Z = torch.full((len(G),len(i)*n_feats), float('nan'))
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

                X, Y, G = env.train()
                Y = Y[:,self.y]
                W = make_weighted(G)

                if self.s2[-1] != n_y*n_persons:
                    Z = Y
                else:
                    i = defaultdict(lambda c= count(0):next(c))
                    I = torch.tensor([[i[g]] for g in G.tolist()]) * n_y + torch.arange(n_y).unsqueeze(0)
                    R = torch.arange(len(Y)).unsqueeze(1)
                    Z = torch.full((len(G),len(i)*len(self.y)), float('nan'))
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

                X, Y, G = env.train()
                Y = Y[:,self.y]
                W = make_weighted(G)

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
            for l in mods: l.eval()

        N = 20
        scores = [ [] for _ in range(N+1) ]
        unchanged_mods_opts = mods_opts

        for j in range(90):

            mods_opts = []
            for mods,opts in unchanged_mods_opts:
                mods = deepcopy(mods)
                opts = deepcopy(opts)
                mods_opts.append([mods,opts])

            X, Y = env.test()
            trn,tst = next(StratifiedShuffleSplit(1,train_size=N/len(X),random_state=j).split(X,Y))
            X = X[np.hstack([trn,tst])]
            Y = Y[np.hstack([trn,tst]),:][:,self.y]

            for mods,opts in mods_opts:
                if not self.pers_rank:
                    opts[-1] = COCOB(mods[-1].parameters()) if opts[-1] else None
                else:
                    targets  = [ n for n, m in mods[-1].named_modules() if isinstance(m,torch.nn.Linear)]
                    config   = peft.LoraConfig(r=self.pers_rank, target_modules=targets)
                    mods[-1] = peft.get_peft_model(mods[-1], config)
                    opts[-1] = COCOB(mods[-1].parameters())

            lrnxs = [[] for _ in range(len(mods_opts))]
            lrnys = [[] for _ in range(len(mods_opts))]
            memss = [[] for _ in range(len(mods_opts))]

            def predict(X):
                preds = torch.tensor(0)
                for mods,_ in mods_opts:
                    [s1,_,s2,_,s3] = mods
                    if s3: preds = preds + torch.sigmoid(s3(s2(s1(X.nan_to_num()))))
                return preds/len(mods_opts)

            def score(X,Y):
                with torch.no_grad():
                    out = {}
                    probs = predict(X)
                    preds = (probs>=.5).float()
                    for i,y in enumerate(self.y):

                        tp = ((preds[:,i]==1) & (Y[:,y]==1)).float().mean().item()
                        tn = ((preds[:,i]==0) & (Y[:,y]==0)).float().mean().item()
                        fp = ((preds[:,i]==1) & (Y[:,y]==0)).float().mean().item()
                        fn = ((preds[:,i]==0) & (Y[:,y]==1)).float().mean().item()

                        out[f"auc{i}"] = roc_auc_score(Y[:,y],probs[:,i])
                        out[f"bal{i}"] = balanced_accuracy_score(Y[:,y],preds[:,i])
                        out[f"sen{i}"] = tp/(tp+fn)
                        out[f"spe{i}"] = tn/(tn+fp)

                        for j in [0,1]:
                            out[f"f1{j}{i}" ] = f1_score(Y[:,y],preds[:,i],pos_label=j)
                            out[f"pre{j}{i}"] = precision_score(Y[:,y],preds[:,i],pos_label=j,zero_division=0)
                            out[f"rec{j}{i}"] = recall_score(Y[:,y],preds[:,i],pos_label=j)

                        out[f"f1m{i}"] = f1_score(Y[:,y],preds[:,i],average='macro')
                        out[f"f1w{i}"] = f1_score(Y[:,y],preds[:,i],average='weighted')

                    return out

            scores[0].append(score(X[N:,:], Y[N:,:]))

            loss = torch.nn.BCEWithLogitsLoss()
            for i in range(N):

                for lrnx,lrny,mems,(mods,opts) in zip(lrnxs,lrnys,memss,mods_opts):
                    [s1,_,s2,_,s3 ] = mods
                    [_,_,_,_,s3opt] = opts

                    x,y = X[i,:], Y[i,:]

                    if self.pers_lrn_cnt:
                        lrnx.append(x)
                        lrny.append(y)

                        if self.pers_mem_cnt: 
                            mems.append([x,y,self.pers_mem_rpt])

                        if len(mems) > self.pers_mem_cnt and self.pers_mem_rcl > rng.random():
                            rng.shuffle(mems, inplace=True)
                            for j in reversed(range(1 if self.pers_mem_rcl < 1 else self.pers_mem_rcl)):
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

                scores[i+1].append(score(X[N:,:], Y[N:,:]))

        for s in scores:
            yield {k:mean([s_[k] for s_ in s]) for k in s[0].keys()}

def make_envs(X, Y, G, R, feats):

    too_short = set(g for g in set(G.tolist()) if (g==G).sum() < 30)
    all_equal = set(g for g in set(G.tolist()) if any(len(set(y.tolist()))==1 for y in Y[g==G].T))

    if any(all_equal): print(f"All Equal, no environment added for {sorted(all_equal)}")

    for rng,g in product(range(R),sorted(set(G.tolist())-all_equal-too_short)):
        try:
            next(StratifiedShuffleSplit(1,random_state=rng).split(X[g==G], Y[g==G]))
            yield MyEnvironment(X[g!=G], Y[g!=G], G[g!=G], X[g==G], Y[g==G], feats, g, rng)
        except ValueError as e:
            if 'The least populated class in y has only 1 member' in str(e): continue
            raise

def _work_items(hrs,scs,lins,bats,peds,locs,init,freq,lmin,lmax,ind):
    for pid in sorted(can_predict["ParticipantId"].drop_duplicates().tolist()):
        sub  = can_predict[can_predict.ParticipantId == pid]
        tss  = sub["SubmissionTimestampUtc"].tolist()
        tzs  = sub["LocalTimeZone"].tolist()
        ys   = sub["ER Interest"].tolist()
        args = [[hrs],[scs],[lins],[bats],[peds],[locs,init,freq,lmin,lmax]]
        yield pid,tss,tzs,ys,args,ind

def _make_xyg3(work_item):
    (pid,ts,tz,ys,args,ind) = work_item
    #hrs, scs, lins, bats, peds, locs
    fs = [
        list(locs1(pid,ts,*args[5])),
        list(lins1(pid,ts,*args[2])),
        list(scs(pid,ts,*args[1])),
        list(hrs(pid,ts,*args[0])),
        list(tims(ts,tz)),
        list(bats(pid,ts,*args[3])),
        list(peds(pid,ts,*args[4]))
    ]

    if ind:
        for f in fs:
            add1(f)

    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    ys = [float(y<np.mean(ys)) for y in ys]
    gs = [pid]*len(ys)
    return xs,ys,gs

def _make_xyg1(work_item):
    (pid,ts,tz,ys,args,ind) = work_item
    #hrs, scs, lins, bats, peds, locs
    fs = [
        list(locs1(pid,ts,*args[5])),
        list(lins1(pid,ts,*args[2])),
        list(tims(ts,tz)),
        list(bats(pid,ts,*args[3])),
        list(peds(pid,ts,*args[4]))
    ]

    if ind:
        for f in fs:
            add1(f)

    xs = [list(chain.from_iterable(feats)) for feats in zip(*fs)]
    ys = [float(y<np.mean(ys)) for y in ys]
    gs = [pid]*len(ys)
    return xs,ys,gs

hour=60*(minute:=60)
envs = []

with ProcessPoolExecutor(max_workers=20) as executor:
    X,Y,G = zip(*executor.map(_make_xyg1, _work_items(30,30,30,3600,30,3600,'geometric',2,1,10,False)))
    X = torch.tensor(list(chain.from_iterable(X))).float()
    Y = torch.tensor(list(chain.from_iterable(Y))).float().unsqueeze(1)
    G = torch.tensor(list(chain.from_iterable(G))).int()
    envs.extend(make_envs(X,Y,G,15,('xyg1',1,2,30,3600,False)))

# with ProcessPoolExecutor(max_workers=20) as executor:
#     X,Y,G = zip(*executor.map(_make_xyg1, _work_items(30,30,30,3600,30,3600,'geometric',2,1,10,True)))
#     X = torch.tensor(list(chain.from_iterable(X))).float()
#     Y = torch.tensor(list(chain.from_iterable(Y))).float().unsqueeze(1)
#     G = torch.tensor(list(chain.from_iterable(G))).int()
#     envs.extend(make_envs(X,Y,G,15,('xyg1',1,2,30,3600,True)))

clear_output()

lrns = [ None ]
vals = [ 
    MyEvaluator(('x',.3,120,'l','r',120,'l','r','-x'), (120,90,'l','r',-1), (90,1), 2, 4, 4, 1, 4, 3, 2, 2, 2, 0, [0], 1, [3]),
]

cb.Experiment(envs,lrns,vals).run(processes=1,quiet=True) #type: ignore