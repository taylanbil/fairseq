# FIXME: remove this file
def metsumm(stepno=''):
    import torch_xla.core.xla_model as xm
    if hasattr(metsumm, 'STEPNO'):
        metsumm.STEPNO += stepno.lower()=="before forward"
    else:
        metsumm.STEPNO = 0
    try:
        import torch_xla.debug.metrics as met
        x = met.metrics_report().split('\n')
        for i, line in enumerate(x):
            if 'CompileTime' in line or 'aten::' in line:
                key = line.split()[-1]
                value = x[i+1].split()[-1]
                xm.master_print('step {}-{}, key {}, value {}'.format(
                    metsumm.STEPNO, stepno, key, value)
                )
    except RuntimeError:
        return
