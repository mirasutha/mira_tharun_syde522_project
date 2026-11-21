from common import *
from sklearn.svm import SVC
out={'linear':[],'poly':[],'rbf':[]}
for C in [0.001,0.01,0.1,1,10]:
    t=time.time(); m=SVC(kernel='linear',C=C,max_iter=2000000).fit(Str,ytr)
    out['linear'].append({'C':C,'val':acc(m,Sva,yva),'train':acc(m,Str,ytr),'nsv':int(m.n_support_.sum()),'time':time.time()-t}); print('linear',C,out['linear'][-1],flush=True)
for d in [2,3,4]:
    for C in [1,10]:
        t=time.time(); m=SVC(kernel='poly',degree=d,coef0=1,C=C).fit(Str,ytr)
        out['poly'].append({'degree':d,'C':C,'val':acc(m,Sva,yva),'train':acc(m,Str,ytr),'nsv':int(m.n_support_.sum()),'time':time.time()-t}); print('poly',d,C,out['poly'][-1],flush=True)
for C in [0.1,1,10,100,1000]:
    for g in [0.001,0.003,0.01,0.03,0.1,0.3,1]:
        t=time.time(); m=SVC(C=C,gamma=g).fit(Str,ytr)
        out['rbf'].append({'C':C,'gamma':g,'val':acc(m,Sva,yva),'train':acc(m,Str,ytr),'nsv':int(m.n_support_.sum()),'time':time.time()-t}); print('rbf',C,g,round(out['rbf'][-1]['val'],3),round(out['rbf'][-1]['time'],1),flush=True)
save('phaseB',out)
