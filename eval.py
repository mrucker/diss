import pandas as pd
import cloudpickle
import torch
import parameterfree

def train_model1(Xs, ys, model, opt, sched, batch=8, epoch=1, device='cpu', autotype=None):
    dataset = torch.utils.data.TensorDataset(Xs,ys)
    loader  = torch.utils.data.DataLoader(dataset,batch_size=batch,pin_memory=(device!='cpu'),drop_last=True,shuffle=True)

    loss = torch.nn.BCEWithLogitsLoss()
    
    for _ in range(epoch):
        for X,y in loader:
            opt.zero_grad()
            X,y = X.to(device),y.to(device)
            l = loss(model(X),y)
            l.backward()
            opt.step()
        if sched: sched.step()
    return model.eval()

def train_model2(Xs, Ys, model, opt, sched, batch=8, epoch=1, device='cpu', autotype=None):
    dataset = torch.utils.data.TensorDataset(Xs,Ys)
    loader  = torch.utils.data.DataLoader(dataset,batch_size=batch,pin_memory=(device!='cpu'),drop_last=True,shuffle=True)

    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.BCEWithLogitsLoss()

    for _ in range(epoch):
        for X,Y in loader:
            opt.zero_grad()
            X,Y = X.to(device),Y.to(device)
            Yhat = model(X)
            l1 = loss1(Yhat[:,:13],Y[:,:13])
            l2 = loss2(Yhat[:,13:],Y[:,13:])
            (l1+l2).backward()
            opt.step()
        if sched: sched.step()
    return model.eval()

def eval1(args):
    
    #Test Pre without SSL
    
    if len(args) == 9:
        model,pid,key,b,e,numthreads,device,autotype,f = args
    else:
        model,pid,key,b,e,numthreads,device,autotype,f = (*args,"sims_features1.csv")

    torch.set_num_threads(numthreads)

    df = pd.read_csv(f'data/{f}')

    X_all = torch.tensor(df.iloc[:,7:].to_numpy())
    y_all = torch.tensor(((df['experience_id'] != 1) & (df['phase_id'] == 1)).astype(int).to_numpy())[:,None]

    X_all = X_all.float()
    y_all = y_all.float()

    X_trn = X_all[(df.participant_id!=pid).tolist()]
    y_trn = y_all[(df.participant_id!=pid).tolist()]
    X_tst = X_all[(df.participant_id==pid).tolist()]
    y_tst = y_all[(df.participant_id==pid).tolist()]

    model = cloudpickle.loads(model).to(device)
    opt   = parameterfree.COCOB(model.parameters())

    model = train_model1(X_trn,y_trn,model,opt,None,b,e,device,autotype=autotype)

    with torch.no_grad():
        scores = torch.nn.Sigmoid()(model(X_tst.to(device))).squeeze().tolist()
        labels = y_tst.squeeze().tolist()

    return key,scores,labels

def eval2(args):

    #Test Pre with SSL
    if len(args) == 9:
        model,pid,key,b,e,numthreads,device,autotype,f = args
    else:
        model,pid,key,b,e,numthreads,device,autotype,f = (*args,"sims_features1.csv")

    torch.set_num_threads(numthreads)

    df = pd.read_csv(f'data/{f}')
    
    X = df.groupby(['participant_id','phase_id','experience_id']).head(n=-1)
    X = X.reset_index(drop=True)
    Y = df.groupby(['participant_id','phase_id','experience_id']).tail(n=-1)
    Y = Y.reset_index(drop=True)

    pids = Y.participant_id
    
    X_all = X.iloc[:,7:]
    Y_all = pd.concat([Y.iloc[:,7:], ((Y['experience_id'] != 1) & (Y['phase_id'] == 1)).astype(int)],axis=1)
    
    X_all = torch.tensor(X_all.to_numpy()).float()
    Y_all = torch.tensor(Y_all.to_numpy()).float()

    X_trn = X_all[(Y.participant_id!=pid).tolist()]
    Y_trn = Y_all[(Y.participant_id!=pid).tolist()]
    X_tst = X_all[(Y.participant_id==pid).tolist()]
    Y_tst = Y_all[(Y.participant_id==pid).tolist()]

    model = cloudpickle.loads(model).to(device)
    opt   = parameterfree.COCOB(model.parameters())

    model = train_model2(X_trn,Y_trn,model,opt,None,b,e,device,autotype=autotype)

    with torch.no_grad():
        scores = torch.nn.Sigmoid()(model(X_tst.to(device))[:,13:]).squeeze().tolist()
        labels = Y_tst[:,13:].squeeze().tolist()

    return key,scores,labels

def eval3(args):

    #Test subjective with SSL
    if len(args) == 9:
        model,pid,key,b,e,numthreads,device,autotype,f = args
    else:
        model,pid,key,b,e,numthreads,device,autotype,f = (*args,"sims_features1.csv")

    torch.set_num_threads(numthreads)

    df2 = pd.read_csv('data/self_reports.csv')
    df2 = df2[df2.phase != 'baseline']
    df2['participant_number'] = df2['PID']
    prt = pd.read_csv('data/participants.csv')
    prt.participant_number = prt.participant_number.str[1:].astype(int)
    prt = prt[['participant_id','participant_number']]
    exp = pd.DataFrame([[1,'alone_video'],[2,'dyad_evaluative'],[3,'group_evaluative'],[4,'dyad_non_evaluative'],[5,'group_non_evaluative']],columns=['experience_id','experience'])
    phs = pd.DataFrame([[1,'anticipatory anxiety'],[2,'experience'],[3,'post-event']],columns=['phase_id','phase'])
    df2 = df2[['participant_number','experience','phase','calm_anx']]
    df2 = pd.merge(pd.merge(pd.merge(df2,exp),phs),prt)
    df2 = df2[['participant_id','experience_id','phase_id','calm_anx']]
    df3 = pd.read_csv(f'data/{f}')
    df4 = pd.merge(df2,df3)
    df4['calm_anx'] = df4['calm_anx'].map({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
    X = df4.groupby(['participant_id','phase_id','experience_id']).head(n=-1)
    X = X.reset_index(drop=True)
    Y = df4.groupby(['participant_id','phase_id','experience_id']).tail(n=-1)
    Y = Y.reset_index(drop=True)
    pids = Y.participant_id
    
    X_all = X.iloc[:,8:]
    Y_all = pd.concat([Y.iloc[:,8:], Y['calm_anx']],axis=1)
    
    X_all = torch.tensor(X_all.to_numpy()).float()
    Y_all = torch.tensor(Y_all.to_numpy()).float()

    X_trn = X_all[(Y.participant_id!=pid).tolist()]
    Y_trn = Y_all[(Y.participant_id!=pid).tolist()]
    X_tst = X_all[(Y.participant_id==pid).tolist()]
    Y_tst = Y_all[(Y.participant_id==pid).tolist()]

    model = cloudpickle.loads(model).to(device)
    opt   = parameterfree.COCOB(model.parameters())

    model = train_model2(X_trn,Y_trn,model,opt,None,b,e,device,autotype=autotype)

    with torch.no_grad():
        scores = torch.nn.Sigmoid()(model(X_tst.to(device))[:,13:]).squeeze().tolist()
        labels = Y_tst[:,13:].squeeze().tolist()

    return key,scores,labels

def eval3b(args):

    #Test subjective without SSL
    if len(args) == 9:
        model,pid,key,b,e,numthreads,device,autotype,f = args
    else:
        model,pid,key,b,e,numthreads,device,autotype,f = (*args,"sims_features1.csv")
    torch.set_num_threads(numthreads)

    df2 = pd.read_csv('data/self_reports.csv')
    prt = pd.read_csv('data/participants.csv')
    df3 = pd.read_csv(f'data/{f}')
    
    df2 = df2[df2.phase != 'baseline']
    df2['participant_number'] = df2['PID']
    prt.participant_number = prt.participant_number.str[1:].astype(int)
    prt = prt[['participant_id','participant_number']]
    exp = pd.DataFrame([[1,'alone_video'],[2,'dyad_evaluative'],[3,'group_evaluative'],[4,'dyad_non_evaluative'],[5,'group_non_evaluative']],columns=['experience_id','experience'])
    phs = pd.DataFrame([[1,'anticipatory anxiety'],[2,'experience'],[3,'post-event']],columns=['phase_id','phase'])
    df2 = df2[['participant_number','experience','phase','calm_anx']]
    df2 = pd.merge(pd.merge(pd.merge(df2,exp),phs),prt)
    df2 = df2[['participant_id','experience_id','phase_id','calm_anx']]
    df4 = pd.merge(df2,df3)
    
    df4['calm_anx'] = df4['calm_anx'].map({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})

    Z = df4
    
    pids = Z.participant_id
    
    X_all = Z.iloc[:,8:]
    Y_all = Z['calm_anx']
    
    X_all = torch.tensor(X_all.to_numpy()).float()
    Y_all = torch.tensor(Y_all.to_numpy()).float()[:,None]

    X_trn = X_all[(Z.participant_id!=pid).tolist()]
    Y_trn = Y_all[(Z.participant_id!=pid).tolist()]
    X_tst = X_all[(Z.participant_id==pid).tolist()]
    Y_tst = Y_all[(Z.participant_id==pid).tolist()]

    model = cloudpickle.loads(model).to(device)
    opt   = parameterfree.COCOB(model.parameters())

    model = train_model1(X_trn,Y_trn,model,opt,None,b,e,device,autotype=autotype)

    with torch.no_grad():
        scores = torch.nn.Sigmoid()(model(X_tst.to(device))).squeeze().tolist()
        labels = Y_tst.squeeze().tolist()

    return key,scores,labels

def eval4(args):
    #Test Pre/Post with SSL

    if len(args) == 9:
        model,pid,key,b,e,numthreads,device,autotype,f = args
    else:
        model,pid,key,b,e,numthreads,device,autotype,f = (*args,"sims_features1.csv")

    torch.set_num_threads(numthreads)

    df = pd.read_csv(f'data/{f}')
    
    X = df.groupby(['participant_id','phase_id','experience_id']).head(n=-1)
    X = X.reset_index(drop=True)
    Y = df.groupby(['participant_id','phase_id','experience_id']).tail(n=-1)
    Y = Y.reset_index(drop=True)

    X = X[(Y['phase_id'] == 2) & (Y['experience_id'] != 1)]
    Y = Y[(Y['phase_id'] == 2) & (Y['experience_id'] != 1)]
    
    pids = Y.participant_id
    
    X_all = X.iloc[:,7:]
    Y_all = pd.concat([Y.iloc[:,7:], (Y['phase_id'] == 1).astype(int)],axis=1)
    
    X_all = torch.tensor(X_all.to_numpy()).float()
    Y_all = torch.tensor(Y_all.to_numpy()).float()

    X_trn = X_all[(Y.participant_id!=pid).tolist()]
    Y_trn = Y_all[(Y.participant_id!=pid).tolist()]
    X_tst = X_all[(Y.participant_id==pid).tolist()]
    Y_tst = Y_all[(Y.participant_id==pid).tolist()]

    model = cloudpickle.loads(model).to(device)
    opt   = parameterfree.COCOB(model.parameters())

    model = train_model2(X_trn,Y_trn,model,opt,None,b,e,device,autotype=autotype)

    with torch.no_grad():
        scores = torch.nn.Sigmoid()(model(X_tst.to(device))[:,13:]).squeeze().tolist()
        labels = Y_tst[:,13:].squeeze().tolist()

    return key,scores,labels

def eval5(args):
    #Test Social with SSL

    model,pid,key,b,e,numthreads,device,autotype,f = args

    torch.set_num_threads(numthreads)

    df = pd.read_csv(f'data/{f}')
    
    X = df.groupby(['participant_id','phase_id','experience_id']).head(n=-1)
    X = X.reset_index(drop=True)
    Y = df.groupby(['participant_id','phase_id','experience_id']).tail(n=-1)
    Y = Y.reset_index(drop=True)

    X = X[(Y['phase_id'] == 2)]
    Y = Y[(Y['phase_id'] == 2)]
    
    pids = Y.participant_id
    
    X_all = X.iloc[:,7:]
    Y_all = pd.concat([Y.iloc[:,7:], (Y['experience_id'] == 1).astype(int)],axis=1)
    
    X_all = torch.tensor(X_all.to_numpy()).float()
    Y_all = torch.tensor(Y_all.to_numpy()).float()

    X_trn = X_all[(Y.participant_id!=pid).tolist()]
    Y_trn = Y_all[(Y.participant_id!=pid).tolist()]
    X_tst = X_all[(Y.participant_id==pid).tolist()]
    Y_tst = Y_all[(Y.participant_id==pid).tolist()]

    model = cloudpickle.loads(model).to(device)
    opt   = parameterfree.COCOB(model.parameters())

    model = train_model2(X_trn,Y_trn,model,opt,None,b,e,device,autotype=autotype)

    with torch.no_grad():
        scores = torch.nn.Sigmoid()(model(X_tst.to(device))[:,13:]).squeeze().tolist()
        labels = Y_tst[:,13:].squeeze().tolist()

    return key,scores,labels
