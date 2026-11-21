# Simple models: ridge/OLS (one-hot least squares), +poly features, logistic regression, perceptron
from common import *
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, Perceptron
from sklearn.preprocessing import PolynomialFeatures
Ytr=onehot(ytr); out={}
def lsacc(m,A,B): return float((CLASSES[m.predict(B).argmax(1)]==yva).mean()), float((CLASSES[m.predict(A).argmax(1)]==ytr).mean())
t=time.time(); m=LinearRegression().fit(Str,Ytr); v,tr=lsacc(m,Str,Sva); out['ols']={'val':v,'train':tr}; print('OLS',v,tr,flush=True)
alphas=[1e-3,1e-2,1e-1,1,10,100,1e3,1e4,1e5]; out['ridge']=[]
for d in [1,2,3]:
    if d==1: A,B=Str,Sva
    else:
        P=PolynomialFeatures(d,include_bias=False); A=P.fit_transform(Str); B=P.transform(Sva)
        sc=StandardScaler().fit(A); A,B=sc.transform(A),sc.transform(B)
    for al in alphas:
        v,tr=lsacc(Ridge(alpha=al).fit(A,Ytr),A,B); out['ridge'].append({'degree':d,'alpha':al,'val':v,'train':tr,'nfeat':A.shape[1]}); print('ridge',d,al,round(v,3),round(tr,3),flush=True)
out['logreg']=[]
for pen in ['l2','l1']:
    for C in [1e-4,1e-3,1e-2,1e-1,1,10,100,1e3]:
        t=time.time()
        m=LogisticRegression(C=C,max_iter=1000) if pen=='l2' else LogisticRegression(C=C,penalty='l1',solver='saga',max_iter=200,tol=1e-3)
        m.fit(Str,ytr); out['logreg'].append({'penalty':pen,'C':C,'val':acc(m,Sva,yva),'train':acc(m,Str,ytr),'time':time.time()-t}); print('logreg',pen,C,round(out['logreg'][-1]['val'],3),flush=True)
out['perceptron']=[]
for ep in [1,2,3,5,10,20,50,100,200]:
    vs,ts=[],[]
    for s in range(5):
        m=Perceptron(max_iter=ep,tol=None,random_state=s).fit(Str,ytr); vs.append(acc(m,Sva,yva)); ts.append(acc(m,Str,ytr))
    out['perceptron'].append({'epochs':ep,'val_mean':float(np.mean(vs)),'val_sd':float(np.std(vs)),'val':vs,'train_mean':float(np.mean(ts))}); print('perc',ep,round(np.mean(vs),3),round(np.std(vs),3),flush=True)
ref=Perceptron(max_iter=50,tol=None,random_state=0,eta0=1.0).fit(Str,ytr).predict(Sva); out['perceptron_eta']=[]
for e in [0.01,0.1,1,10,100]:
    p=Perceptron(max_iter=50,tol=None,random_state=0,eta0=e).fit(Str,ytr).predict(Sva); out['perceptron_eta'].append({'eta0':e,'val':float((p==yva).mean()),'identical_to_eta1':bool((p==ref).all())}); print('eta',e,out['perceptron_eta'][-1],flush=True)
save('phaseA',out)
