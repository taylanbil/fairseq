# FIXME: remove this file
from datetime import datetime
def metsumm(stepno=''):
    if hasattr(metsumm, 'STEPNO'):
        metsumm.STEPNO += stepno.lower().startswith("STEP")
    else:
        metsumm.STEPNO = 0
    try:
        import torch_xla.debug.metrics as met
        import torch_xla.core.xla_model as xm
        if not xm.is_master_ordinal(local=False):
            return
        x = met.metrics_report().split('\n')
        prdate = True
        for i, line in enumerate(x):
            if 'CompileTime' in line or 'aten::' in line:
                key = line.split()[-1]
                value = x[i+1].split()[-1]
                prdate_, prdate = prdate, False
                print(
                    'step {}-{}, {}key {}, value {}'.format(
                        metsumm.STEPNO, stepno,
                        '{} - '.format(datetime.now()) if prdate_ else '',
                        key, value
                    )
                )
    except RuntimeError:
        return
