"""Full-shape sparse diagnostic weights. Never substitute these for real capture."""
from pathlib import Path
from common import K,N,WEIGHT_BYTES,record,write_json


def prepare(root):
    import numpy as np
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    specs=[('zero',()),('tile_tie',(63,64)),('partition_tie',(64,53760)),('last_vocab',(N-1,)),
           ('rounded_tie',(63,64)),('rounded_up',(63,64)),('row_identity',()),
           ('all_negative',()),('overflow_inf',(63,64)),('nan_first',(63,64)),('signed_zero_underflow',(1,))]
    cases=[]
    for name,ids in specs:
        folder=root/name; folder.mkdir(); wpath=folder/'head.bin'
        with wpath.open('wb') as f: f.truncate(WEIGHT_BYTES)
        w=np.memmap(wpath,mode='r+',dtype='<f2',shape=(N,K))
        x=np.zeros((15,K),dtype='<f2'); x[:,0]=1
        expected=[0]*15
        if name=='zero': x[:,::2]=-0.0
        elif name in ('tile_tie','partition_tie','last_vocab'):
            for token in ids: w[token,0]=3
            expected=[min(ids)]*15
        elif name in ('rounded_tie','rounded_up'):
            x[:,1]=2**-11 if name=='rounded_tie' else 2**-10
            w[63,0]=w[64,0]=1; w[64,1]=0.5 if name=='rounded_tie' else 1
            expected=[63 if name=='rounded_tie' else 64]*15
        elif name=='row_identity':
            x[:]=0
            for row in range(15): x[row,row]=1; w[row+1,row]=2
            expected=list(range(1,16))
        elif name=='all_negative':
            # Every vocabulary row is negative, so a zero-initialized max fails.
            w[:,0]=-1; w[N-1,0]=-0.5; expected=[N-1]*15
        elif name=='overflow_inf':
            x[:,0]=2; w[63,0]=w[64,0]=65504; expected=[63]*15
        elif name=='nan_first':
            w[63,0]=np.nan; w[64,0]=np.inf; expected=[63]*15
        elif name=='signed_zero_underflow':
            x[:,0]=2**-24; w[1,0]=-0.5
        w.flush(); del w
        x.tofile(folder/'hidden.bin')
        cases.append(dict(name=name,m=15,source='synthetic-full-vocabulary',expected_ids=expected,
                          hidden=record(folder/'hidden.bin',root),weight=record(wpath,root)))
    write_json(root/'manifest.json',dict(status='PREPARED_NOT_EXECUTED',cases=cases))
    return cases
