import torch
import parameterfree

def evaluator(args):

    exp,t,device,seed = args

    torch.set_num_threads(t)
    torch.manual_seed(seed)

    b,e   = exp._b,exp._e
    data  = exp.load_data()
    model = exp.make_model().to(device)
    opt   = parameterfree.COCOB(model.parameters())

    X_trn,Y_trn,_,X_tst,Y_tst,_ = data

    torch_dataset = torch.utils.data.TensorDataset(X_trn,Y_trn)
    torch_loader  = torch.utils.data.DataLoader(torch_dataset,batch_size=b,pin_memory=(device!='cpu'),drop_last=True,shuffle=True)

    for _ in range(e):
        for X,Y in torch_loader:

            Ystar = Y.to(device)
            Yhat  = model(X.to(device))

            opt.zero_grad()
            exp.loss(Yhat,Y).backward()
            opt.step()

    return exp.test_model(model.eval(), X_tst.to(device),Y_tst.to(device))
