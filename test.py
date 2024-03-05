import coba as cb

from learners import EMT, EmtStackedLearner

import parameterfree #https://github.com/bremen79/parameterfree
import torch

cb.CobaContext

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, normed=True):
        super().__init__()

        input_norm   = [torch.nn.LayerNorm(in_features)] if normed else []
        output_layer = torch.nn.Linear(in_features=in_features, out_features=out_features)
        
        self.layers  = torch.nn.Sequential(*input_norm,output_layer)

    def forward(self, Xs):
        return self.layers(Xs)

class ResNet(torch.nn.Module):

    def __init__(self, in_features, out_features, hidden=3, width=None, normed=True):
        super().__init__()

        class PreActivationResidualBlock(torch.nn.Module):
            #https://arxiv.org/abs/1603.05027
            def __init__(self, in_width, normed=True) -> None:
                super().__init__()

                def not_normed_layer():
                    yield torch.nn.LeakyReLU()
                    yield torch.nn.Linear(in_features=in_width, out_features=in_width)

                def normed_layer():
                    yield torch.nn.LayerNorm(in_width)
                    yield from not_normed_layer()

                layer = normed_layer if normed else not_normed_layer        
                self.layers = torch.nn.Sequential(*layer(),*layer())

            def forward(self, Xs):
                return Xs+self.layers(Xs)

        width  = width or in_features

        input_norm    = [torch.nn.LayerNorm(in_features)] if normed else []
        input_layer   = [torch.nn.Linear(in_features=in_features, out_features=width)]
        hidden_layers = [PreActivationResidualBlock(width,normed) for _ in range(hidden)]
        output_layer  = torch.nn.Linear(in_features=width, out_features=out_features)

        self.layers = torch.nn.Sequential(*input_norm,*input_layer,*hidden_layers,output_layer)

    def forward(self, Xs):
        return self.layers(Xs)

class Mlp(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden=3, width=None, normed=True):
        super().__init__()

        def not_normed_layers(_in_width,_out_width):
            yield torch.nn.LeakyReLU()
            yield torch.nn.Linear(in_features=_in_width, out_features=_out_width)

        def normed_layers(_in_width,_out_width):
            yield torch.nn.LayerNorm(_in_width)
            yield from not_normed_layers(_in_width,_out_width)

        width  = width or in_features
        layers = normed_layers if normed else not_normed_layers

        input_norm    = [torch.nn.LayerNorm(in_features)] if normed else []
        input_layer   = [torch.nn.Linear(in_features=in_features, out_features=width)   ]
        hidden_layers = [torch.nn.Sequential(*layers(width,width)) for _ in range(hidden)]
        output_layer  = torch.nn.Linear(in_features=width, out_features=out_features)

        self.layers = torch.nn.Sequential(*input_norm,*input_layer,*hidden_layers,output_layer)

    def forward(self, Xs):
        return self.layers(Xs)

class NeuralSquareCbLearner1:
    def __init__(self):
        self.loss   = torch.nn.MSELoss(reduction='none')
        self.opt    = None
        self.gamma  = 1000
        self._rng   = cb.CobaRandom(1)
        self._lossm = [0,0]

    def define(self,context,action):
        torch.manual_seed(1)
        self.fhat  = Linear(len(context+action),1)
        self.opt   = torch.optim.Adam(self.fhat.parameters(),lr=.1)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, 1, .99)

    def predict(self, context, actions):
        context = context or []
        if not self.opt: self.define(context,actions[0])

        with torch.no_grad():
            mu = len(actions)
            X  = torch.tensor([context+action for action in actions])
            y  = self.fhat(X)

            rvals       = torch.reshape(y, (1,-1))
            rmaxs,ridxs = rvals.max(axis=1,keepdim=True)
            rgaps       = rmaxs-rvals

            probs = 1/(mu+self.gamma*rgaps)
            probs[range(1),ridxs.squeeze()] += 1-probs.sum(axis=1)

        return self._rng.choicew(actions,probs[0].tolist())

    def learn(self, context, action, reward, score):
        context = context or []
        if not self.opt: self.define(context,action)

        self.opt.zero_grad()

        reward      = torch.tensor([reward],dtype=torch.float32)
        pred_reward = self.fhat(torch.tensor(context+action))
        loss_mean   = self.loss(pred_reward, reward).mean()

        loss_mean.backward()
        self.opt.step()
        self.sched.step()

        #n = self._lossm[1]
        #self._lossm[0] = (n/(n+1))*self._lossm[0] + (1/(n+1)) * loss_mean.item()
        #self._lossm[1]+=1
        #print(self._lossm[0])

class NeuralSquareCbLearner2:
    def __init__(self):
        self.loss   = torch.nn.MSELoss(reduction='none')
        self.opt    = None
        self.gamma  = 1000
        self._rng   = cb.CobaRandom(1)
        self._lossm = [0,0]

    def define(self,context,action):
        torch.manual_seed(1)
        self.fhat  = Mlp(len(context+action),1)
        self.opt   = parameterfree.COCOB(self.fhat.parameters())

    def predict(self, context, actions):
        context = context or []
        if not self.opt: self.define(context,actions[0])

        with torch.no_grad():
            mu = len(actions)
            X  = torch.tensor([context+action for action in actions])
            y  = self.fhat(X)

            rvals       = torch.reshape(y, (1,-1))
            rmaxs,ridxs = rvals.max(axis=1,keepdim=True)
            rgaps       = rmaxs-rvals

            probs = 1/(mu+self.gamma*rgaps)
            probs[range(1),ridxs.squeeze()] += 1-probs.sum(axis=1)

        return self._rng.choicew(actions,probs[0].tolist())

    def learn(self, context, action, reward, score):
        context = context or []
        if not self.opt: self.define(context,action)

        self.opt.zero_grad()

        reward      = torch.tensor([reward],dtype=torch.float32)
        pred_reward = self.fhat(torch.tensor(context+action))
        loss_mean   = self.loss(pred_reward, reward).mean()

        loss_mean.backward()
        self.opt.step()

        #n = self._lossm[1]
        #self._lossm[0] = (n/(n+1))*self._lossm[0] + (1/(n+1)) * loss_mean.item()
        #self._lossm[1]+=1
        #print(self._lossm[0])

class NeuralSquareCbLearner3:
    def __init__(self):
        self.loss   = torch.nn.MSELoss(reduction='none')
        self.opt    = None
        self.gamma  = 1000
        self._rng   = cb.CobaRandom(1)
        self._lossm = [0,0]

    def define(self,context,action):
        torch.manual_seed(1)
        self.fhat  = ResNet(len(context+action),1)
        self.opt   = parameterfree.COCOB(self.fhat.parameters())

    def predict(self, context, actions):
        context = context or []
        if not self.opt: self.define(context,actions[0])

        with torch.no_grad():
            mu = len(actions)
            X  = torch.tensor([context+action for action in actions])
            y  = self.fhat(X)

            rvals       = torch.reshape(y, (1,-1))
            rmaxs,ridxs = rvals.max(axis=1,keepdim=True)
            rgaps       = rmaxs-rvals

            probs = 1/(mu+self.gamma*rgaps)
            probs[range(1),ridxs.squeeze()] += 1-probs.sum(axis=1)

        return self._rng.choicew(actions,probs[0].tolist())

    def learn(self, context, action, reward, score):
        context = context or []
        if not self.opt: self.define(context,action)

        self.opt.zero_grad()

        reward      = torch.tensor([reward],dtype=torch.float32)
        pred_reward = self.fhat(torch.tensor(context+action))
        loss_mean   = self.loss(pred_reward, reward).mean()

        loss_mean.backward()
        self.opt.step()

        #n = self._lossm[1]
        #self._lossm[0] = (n/(n+1))*self._lossm[0] + (1/(n+1)) * loss_mean.item()
        #self._lossm[1]+=1
        #print(self._lossm[0])

if __name__ == "__main__":

    lrn1 = cb.VowpalEpsilonLearner(features=[1,"a","xa"])

    lrn = [NeuralSquareCbLearner1(),NeuralSquareCbLearner2(),NeuralSquareCbLearner3(), lrn1]

    #env = cb.Environments.from_linear_synthetic(10_000, n_context_features=3, n_action_features=5, reward_features=['a','xa'])
    env = cb.Environments.from_mlp_synthetic(10_000, n_context_features=3, n_action_features=5)

    cb.Experiment(env,lrn).run(processes=3).plot_learners()
