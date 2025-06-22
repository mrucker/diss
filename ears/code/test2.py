import coba as cb
import torch
from pathlib import Path
from collections import defaultdict, Counter
from itertools import islice, chain, count, product, repeat
from contextlib import nullcontext

data_dir = "./ears/data"

import coba as cb

import warnings

import time
import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler, QuantileTransformer, MinMaxScaler, StandardScaler, Binarizer
from sklearn.feature_selection import  mutual_info_classif, f_classif, GenericUnivariateSelect
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, GridSearchCV, LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

import torch
from parameterfree import COCOB

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

from concurrent.futures import ProcessPoolExecutor

import torch
import torch.utils
import torch.utils.data
import coba as cb
from typing import Tuple, Optional

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

df = pd.read_csv(f"{data_dir}/all_features_1h_v3.csv")

G  = df["id_participant"].to_numpy()
X1 = df[[c for c in df.columns if c.startswith("acc_")]].to_numpy()
X2 = df[[c for c in df.columns if c.startswith("acc_") or c.startswith("gps_") or c.startswith("motion_")]].to_numpy()
Y1 = df["ER_desire"].astype(float).to_numpy()
Y2 = (df["INT_availability"] == "yes").astype(float).to_numpy()

no_na = ~(np.isnan(Y1) | np.isnan(Y2))

G  = G [no_na]
X1 = X1[no_na]
X2 = X2[no_na]
X3 = X2.copy()
X4 = X2.copy()
Y1 = np.expand_dims(Y1[no_na],axis=1)
Y2 = np.expand_dims(Y2[no_na],axis=1)

for g in set(G):
    Y1[G == g] = Binarizer(threshold=np.mean(Y1[G == g].squeeze())).fit_transform(Y1[G == g])
    X1[G == g] = StandardScaler().fit_transform(X1[G == g])
    X2[G == g] = StandardScaler().fit_transform(X2[G == g])
    X3[G == g] = StandardScaler().fit_transform(X3[G == g])
    X4[G == g] = StandardScaler().fit_transform(X4[G == g])

X3 = np.concatenate([X3, np.expand_dims((df["Platform"] == "Android").astype(float).to_numpy(),1)[no_na]],axis=1)
X3 = np.concatenate([X3, np.expand_dims((df["Platform"] == "iPhone" ).astype(float).to_numpy(),1)[no_na]],axis=1)

X4 = np.concatenate([X4, np.expand_dims((df["Platform"] == "Android").astype(float).to_numpy(),1)[no_na]],axis=1)
X4 = np.concatenate([X4, np.expand_dims((df["Platform"] == "iPhone" ).astype(float).to_numpy(),1)[no_na]],axis=1)
X4 = np.concatenate([X4, np.expand_dims((df["tag"] == "evening" ).astype(float).to_numpy(),1)[no_na]],axis=1)
X4 = np.concatenate([X4, np.expand_dims((df["tag"] == "morning" ).astype(float).to_numpy(),1)[no_na]],axis=1)
X4 = np.concatenate([X4, np.expand_dims((df["tag"] == "afternoon" ).astype(float).to_numpy(),1)[no_na]],axis=1)

###################################################################################################

import coba as cb

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
        raise Exception("Bad Layer")

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
        rng_indexes = cb.CobaRandom(self.params['rng']).shuffle(range(len(self.train_X)))
        return self.train_X[rng_indexes,:], self.train_Y[rng_indexes,:], self.train_G[rng_indexes]

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
        from sklearn.metrics import roc_auc_score
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

        if self.ws_steps0:
            if self.s2[-1] == -1: self.s2 = (*(self.s2)[:-1], len(set(env.train()[2].tolist()))*len(self.y))
            if self.s3[ 0] == -1: self.s3 = (len(set(env.train()[2].tolist()))*len(self.y), *(self.s3)[1:])

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

            if self.dae_steps:
                opts = list(filter(None,[s1opt,saopt]))
                X,_,G = env.train()
                W = make_weighted(G)

                torch_dataset = torch.utils.data.TensorDataset(X,W)
                torch_loader  = torch.utils.data.DataLoader(torch_dataset,batch_size=8,drop_last=True,shuffle=True)

                loss = torch.nn.L1Loss()
                for _ in range(self.dae_steps):
                    for (_X,_w) in torch_loader:
                        for o in opts: o.zero_grad()
                        loss(sa(s1(_X.nan_to_num()))[~_X.isnan()],_X[~_X.isnan()]).backward()
                        for o in opts: o.step()

            if self.ws_steps0:
                opts = list(filter(None,[s1opt,s2opt,sbopt]))
                for o in opts: o.zero_grad()

                X, Y, G = env.train()
                Y = Y[:,self.y]
                W = make_weighted(G)

                if self.s2[-1] in [1,2]:
                    Z = Y
                else:
                    i = defaultdict(lambda c= count(0):next(c))
                    I = torch.tensor([[i[g]] for g in G.tolist()]) + torch.arange(len(self.y)).unsqueeze(0)
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

        N = 40
        scores = [ [] for _ in range(N+1) ]
        unchanged_mods_opts = deepcopy(mods_opts)

        for i in range(90):

            mods_opts = deepcopy(unchanged_mods_opts)

            X, Y = env.test()
            trn,tst = next(StratifiedShuffleSplit(1,train_size=N/len(X),random_state=i).split(X,Y))
            X = X[np.hstack([trn,tst])]
            Y = Y[np.hstack([trn,tst]),:][:,self.y]

            for mods,opts in mods_opts:
                if not self.pers_rank:
                    opts[-1] = COCOB(mods[-1].parameters())
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
                for mods, _ in mods_opts:
                    [s1,_,s2,_,s3] = mods
                    preds = preds + torch.sigmoid(s3(s2(s1(X.nan_to_num()))))
                return preds/len(mods_opts)

            def score(X,Y):
                with torch.no_grad():
                    return [ roc_auc_score(Y[:,i],predict(X)[:,i]) for i,y in enumerate(self.y)]

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
                            x,y,n = mems[j]
                            lrnx.append(x)
                            lrny.append(y)
                            if n == 1: mems.pop(j)
                            else: mems[j] = [x,y,n-1]

                    if self.pers_lrn_cnt and len(lrnx) >= self.pers_lrn_cnt:
                        x = torch.stack(lrnx[:self.pers_lrn_cnt])
                        y = torch.stack(lrny[:self.pers_lrn_cnt])
                        if s3opt: s3opt.zero_grad()
                        loss(s3(s2(s1(x.nan_to_num()))),y).backward()
                        if s3opt: s3opt.step()
                        del lrnx[:self.pers_lrn_cnt]
                        del lrny[:self.pers_lrn_cnt]

                scores[i+1].append(score(X[N:,:], Y[N:,:]))

        for s in scores:
            yield { f'auc{i}': auc for i,auc in zip(self.y,torch.tensor(s).mean(dim=0).tolist()) }

def make_envs(X, Y, G, R):
    X, Y, G = torch.tensor(X).float(), torch.tensor(Y).float(), torch.tensor(G)

    too_short = set(g for g in set(G.tolist()) if (g==G).sum() < 50)
    all_equal = set(g for g in set(G.tolist()) if any(len(set(y.tolist()))==1 for y in Y[g==G].T))

    if any(all_equal): print(f"All Equal, no environment added for {sorted(all_equal)}")

    for rng,g in product(range(R),sorted(set(G.tolist())-all_equal-too_short)):
        try:
            next(StratifiedShuffleSplit(1,random_state=rng).split(X[g==G], Y[g==G]))
            yield MyEnvironment(X[g!=G], Y[g!=G], G[g!=G], X[g==G], Y[g==G], 'rest', g, rng)
        except ValueError as e:
            if 'The least populated class in y has only 1 member' in str(e): continue
            raise

w = 30
r = lambda w:  ['l', 'r', w, 'l', 'r', w]

lrns = [ None ]
envs = list(make_envs(X3,np.hstack([Y2]),G,1))
vals = lambda x: [
    MyEvaluator((len(x),.3,90,'l','r',len(x)), (90,90,'l','r',-1), (90,1), 1, 1, 4, 1, 2, 3, 10, 2, 2, 0, [  0], 1, weighted=[]),
    MyEvaluator((len(x),.3,90,'l','r',len(x)), (90,90,'l','r',-1), (90,1), 1, 1, 1, 1, 2, 3, 10, 2, 2, 0, [  0], 1, weighted=[2]),
]

cb.Experiment(envs,lrns,vals(X3[0])).run(processes=1,quiet=True)