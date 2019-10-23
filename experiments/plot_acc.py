# plot dimension results
import pickle
from os.path import join

import plotly.graph_objs as go
from plotly.offline import plot

from skopt import load

emb_fname = 'glove.6B.300d' # or 'random'
dims = [4,8,16,32,64,128]
sc_accs,ova_accs = [],[]
for e in ['sc','ova']:
    for dim in dims:
        results_fname = '_'.join(['results', emb_fname, str(dim)])
        results = load(join(e+'_acc_dim',results_fname)+'.pkl')
        if e == 'sc':
            sc_accs.append(-results.fun)
        else:
            ova_accs.append(-results.fun)


results_trace = []
results_trace.append(
    go.Scatter(
        x=dims,
        y=sc_accs,
        mode='lines+markers',
        name='SC'
    )
)
results_trace.append(
    go.Scatter(
        x=dims,
        y=ova_accs,
        mode='lines+markers',
        name='OVA'
    )
)
plot({
    "data" : results_trace,
    "layout": go.Layout(),
},filename='acc_dim.html')