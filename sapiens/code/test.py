import coba as cb
from itertools import product
from sklearn.metrics import roc_auc_score
import pandas as pd

def balanced_accuracy_scores(y_true,y_scores):
    from operator import itemgetter
    from itertools import groupby

    nclass = len(set(y_true))
    n0,n1  = len(y_true)-sum(y_true),sum(y_true)
    c0,c1  = 0,n1
    n0,n1  = (n0*nclass or 1), (n1*nclass or 1)

    yield (0,(c0/(n0 or 1)+c1/(n1 or 1)))

    for t,g in groupby(sorted(zip(y_scores,y_true)),key=itemgetter(0)):
        ls = list(map(itemgetter(1),g))
        c0 += len(ls)-sum(ls)
        c1 -= sum(ls)
        yield (t,(c0/n0+c1/n1))

def moving_average(values, span=None, weights=None):
    from operator import truediv, mul, sub
    from itertools import accumulate, tee, repeat, chain, count
    assert weights == 'exp' or weights == None or len(weights)==len(values)

    if weights=='exp':
        #exponential moving average identical to Pandas' df.ewm(span=span).mean()
        alpha = 2/(1+span)
        cumwindow  = list(accumulate(values          , lambda a,v: v + (1-alpha)*a))
        cumdivisor = list(accumulate([1.]*len(values), lambda a,v: v + (1-alpha)*a))
        return map(truediv, cumwindow, cumdivisor)

    elif span == 1:
        return values

    elif span is None or span >= len(values):
        values  = accumulate(values) if not weights else accumulate(map(mul,values,weights))
        weights = count(1)           if not weights else accumulate(weights)
        return map(truediv, values, weights)

    else:
        v1,v2   = tee(values    if not weights else map(mul,values,weights),2)
        weights = repeat(1) if not weights else weights

        values  = accumulate(map(sub, v1     , chain(repeat(0,span),v2     )))
        weights = accumulate(map(sub, weights, chain(repeat(0,span),weights)))

        return map(truediv,values,weights)

f = 10

res = cb.Result.from_file("./sapiens/notebooks/out7.log").filter_fin(n=f)

out  = res.interactions.to_pandas()
envs = res.environments.to_pandas()
lrns = res.learners.to_pandas()
vals = res.evaluators.to_pandas()
out = pd.merge(pd.merge(out,envs),vals)

def test(o):
    keep = [pid for pid in set(o['pid']) if len(set(o[o.pid == pid]["true"])) > 1]
    o = o[o.pid.isin(keep)]
    if len(o) == 0: return float('nan')
    return round(100*roc_auc_score(o["true"].tolist(),o["pred"].tolist()),2)
    #_,Z = zip(*balanced_accuracy_scores(o['true'].tolist(),o['pred'].tolist()))
    #return round((max(Z)*100),2)

pd.set_option('display.max_rows', 1000) 

out = out[out.pid.isin([413,415]) & (out['index'] == 10)]

o = out.groupby(['specs1','specs2','specs3','neg','ws','ssl','index'])[['pid','true','pred']].apply(test).reset_index()
o.pivot(index=['specs1','specs2','specs3','neg','ws','ssl'],columns='index',values=0)
#o.groupby(['specs1','specs2','specs3','neg','ws','ssl']).mean()