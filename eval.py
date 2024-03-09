import pandas as pd
import cloudpickle
import torch
import parameterfree

def train_model(Xs, ys, model, opt, sched, loss, batch=8, epoch=1, device='cpu',autotype=None):
    dataset = torch.utils.data.TensorDataset(Xs,ys)
    loader  = torch.utils.data.DataLoader(dataset,batch_size=batch,pin_memory=(device!='cpu'),drop_last=True,shuffle=True)
    for _ in range(epoch):
        for X,y in loader:
            opt.zero_grad()
            X,y = X.to(device),y.to(device)
            if not autotype:
                l = loss(model(X),y)
            else:
                with torch.autocast(device_type=device,dtype=autotype):
                    l = loss(model(X),y)
            l.backward()
            opt.step()
        if sched: sched.step()
    return model.eval()

def eval(args):
    
    model,pid,key,b,e,numthreads,device,autotype = args

    torch.set_num_threads(numthreads)

    df = pd.read_csv('sims_features.csv')

    X_all = torch.tensor(df.iloc[:,7:].to_numpy())
    y_all = torch.tensor(((df['experience_id'] != 1) & (df['phase_id'] == 1)).astype(int).to_numpy())[:,None]

    X_all = X_all.float()
    y_all = y_all.float()

    X_trn = X_all[df.participant_id!=pid]
    y_trn = y_all[df.participant_id!=pid]
    X_tst = X_all[df.participant_id==pid]
    y_tst = y_all[df.participant_id==pid]

    model   = cloudpickle.loads(model).to(device)
    loss  = torch.nn.BCEWithLogitsLoss()
    opt   = parameterfree.COCOB(model.parameters())

    model = train_model(X_trn,y_trn,model,opt,None,loss,b,e,device,autotype=autotype)

    with torch.no_grad():
        scores = torch.nn.Sigmoid()(model(X_tst.to(device))).squeeze().tolist()
        labels = y_tst.squeeze().tolist()

    return key,scores,labels