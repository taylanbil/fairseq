
# FIXME
def metsumm(w=''):
    import torch_xla.debug.metrics as met
    m = met.metrics_report().split('\n')
    print('-'*30)
    for i, line in enumerate(m):
        if 'CompileTime' in line or 'aten::' in line:
            key = line.strip().split()[-1]
            value = m[i+1].strip().split()[-1]
            print('@ {}, {} = {}'.format(w, key, value))
            print('-'*30)

def metd():
    import torch_xla.debug.metrics as met
    from collections import Counter
    x = met.metrics_report().split('\n')
    d = {}
    for i, line in enumerate(x):
        if 'Counter:' in line:
            k = line.split(' ')[-1]
            v = x[i+1].split(' ')[-1]
            d[k] = int(v)
    return Counter(d)


def metdelta(d2, d1):
    from collections import Counter
    d2, d1 = Counter(d2), Counter(d1)
    import pandas as pd
    x = pd.DataFrame()
    x['after'] = pd.Series(d2)
    x['before'] = pd.Series(d2)
    x.fillna(0, inplace=True)
    x['delta'] = x['after'] - x['before']
    return x.sort_values(by='delta')
