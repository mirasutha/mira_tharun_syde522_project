import numpy as np, pandas as pd, time, json, warnings, os, sys
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
ROOT='/home/claude/proj'
df=pd.read_csv('/mnt/user-data/uploads/letter-recognition.csv'); df.columns=[c.strip() for c in df.columns]
y_all=df['letter'].to_numpy(dtype=str); X_all=df.drop(columns='letter').values.astype(float)
FEATS=list(df.columns[1:]); CLASSES=np.array(sorted(set(y_all.tolist())))
Xpool,ypool=X_all[:16000],y_all[:16000]          # UCI convention: first 16,000 train pool
Xtest,ytest=X_all[16000:],y_all[16000:]          # last 4,000 test (touched once, in phase D)
perm=np.random.RandomState(0).permutation(16000)
itr,iva=perm[:12000],perm[12000:]                # 12k train / 4k validation for all tuning
Xtr,ytr,Xva,yva=Xpool[itr],ypool[itr],Xpool[iva],ypool[iva]
_pool=set(map(tuple,Xpool)); NOLEAK=np.array([tuple(r) not in _pool for r in Xtest])
def scaled(Xfit,*Xs):
    sc=StandardScaler().fit(Xfit); return [sc.transform(Xfit)]+[sc.transform(x) for x in Xs]
Str,Sva=scaled(Xtr,Xva)
def onehot(y): return (y[:,None]==CLASSES[None,:]).astype(float)
def save(name,obj): json.dump(obj,open(f'{ROOT}/results/{name}.json','w'),indent=1,default=float)
def load(name): return json.load(open(f'{ROOT}/results/{name}.json'))
def acc(m,A,y): return float((m.predict(A)==y).mean())
