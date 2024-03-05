from copy import deepcopy
from typing import Tuple, Sequence, Mapping, Hashable, Any

from coba.safety import SafeLearner
from coba.primitives import Learner, Namespaces, Context, Actions, Action
from coba.random import CobaRandom
from coba.learners import VowpalMediator, VowpalLearner

class EMT:

    def __init__(self, split:int = 100, scorer:str="self_consistent_rank", bound:int=0, features: Sequence[str]=[1,'a','xa']) -> None:

        self._params = {'split':split, 'scorer':scorer, 'bound':bound, 'features':features}

        feat_args = []
        if 1   not in features: feat_args.append('--noconstant')
        if 'a' not in features: feat_args.append('--ignore_linear a')
        if 'x' not in features: feat_args.append('--ignore_linear x')
        feat_args += [ f'--interactions {f}' for f in features if f not in {1, 'a', 'x'} ]

        vw_args = [
            "--emt",
            f"--emt_tree {bound}",
            f"--emt_leaf {split}",
            f"--emt_scorer {scorer}",
            f"--emt_router {'eigen'}",
            f"-b {26}",
            "--min_prediction 0",
            "--max_prediction 3",
            "--coin",
            "--initial_weight 0",
            *feat_args,
            '--quiet',
            '--random_seed 1337'
        ]

        self._vw = VowpalMediator()
        self._vw_args = ' '.join(vw_args)

    @property
    def params(self) -> Mapping[str,Any]:
        return self._params
    
    def predict(self, X: Mapping) -> int:
        if not self._vw.is_initialized: self._vw.init_learner(self._vw_args, label_type=2)
        return int(self._vw.predict(self._vw.make_example(X, None)))

    def learn(self, X: Mapping, y: int, weight: float):
        if not self._vw.is_initialized: self._vw.init_learner(self._vw_args, label_type=2)
        self._vw.learn(self._vw.make_example(X, f"{int(y)} {weight}"))

class EmtLearner:

    def __init__(self, emt: EMT, epsilon: float = 0.05) -> None:
        assert 0 <= epsilon and epsilon <= 1
        self._epsilon = epsilon
        self._emt     = deepcopy(emt)
        self._rng     = CobaRandom(1)

    @property
    def params(self) -> Mapping[str,Any]:
        return { 'family': 'Eigen', 'e':self._epsilon, **self._emt.params }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        r_star, A_star = -float('inf'), []
        for action in actions:
            reward = self._emt.predict({'x':context,'a':action})
            if reward > r_star: r_star, A_star = reward, []
            if reward == r_star: A_star.append(action)

        min_p = self._epsilon/len(actions)
        grd_p = (1-self._epsilon)/len(A_star)
        pmf   = [ grd_p+min_p if a in A_star else min_p for a in actions ]

        return self._rng.choicew(actions, pmf)

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""
        self._emt.learn({'x':context,'a':action}, reward, 1)

class EmtStackedLearner:

    def __init__(self, emt: EMT, learner: Learner) -> None:
        self._emt  = deepcopy(emt)
        self._lrn  = SafeLearner(deepcopy(learner))
        self._isvw = isinstance(learner,VowpalLearner)

    @property
    def params(self) -> Mapping[str,Any]:
        params = {**self._emt.params, **self._lrn.params, 'family': 'EigenStacked'}
        params.pop('args',None)
        return params
    
    def _inner_actions(self, context: Context, actions: Actions) -> Actions:
        eigen_pred  = lambda a: float(self._emt.predict({'x':context,'a':a}))
        if self._isvw:
            #I tested adding the memory to the 'a' namespace directly and it didn't work as well
            return [Namespaces(m=eigen_pred(action),a=action) for action in actions]
        else:
            return [[eigen_pred(action), *action] for action in actions]

    def score(self, context: Context, actions: Actions, action: Action) -> float:
        inner_actions = self._inner_actions(context,actions)
        inner_action  = inner_actions[actions.index(action)]
        return self._lrn.score(context,inner_actions,inner_action)

    def predict(self, context: Context, actions: Actions) -> Tuple[Action,float]:
        """Choose which action index to take."""
        
        outer_actions = actions
        inner_actions = self._inner_actions(context,actions)

        inner_action,inner_score,inner_kwargs = self._lrn.predict(context,inner_actions)
        outer_action = outer_actions[inner_actions.index(inner_action)]

        return outer_action, inner_score, {'inner_action': inner_action, 'inner_kwargs': inner_kwargs}

    def learn(self, context: Hashable, action: Hashable, reward: float, score: float, inner_action=None, inner_kwargs={}) -> None:
        """Learn about the result of an action that was taken in a context."""

        inner_action = inner_action or self._inner_actions(context,[action])[0]
        
        self._emt.learn({'x':context,'a':action}, reward, 1)
        self._lrn.learn(context, inner_action, reward, score, **inner_kwargs)
