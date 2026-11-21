# training-set-size curves (validation accuracy), best config per family
from common import *
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
best=load('best'); out={}
sizes=[250,500,1000,2000,4000,8000,12000]
P=PolynomialFeatures(int(best['ridge_poly']['degree']),include_bias=False)
Ptr=P.fit_transform(Str); Pva=P.transform(Sva); sc=StandardScaler().fit(Ptr); Ptr,Pva=sc.transform(Ptr),sc.transform(Pva)
for n in sizes:
    r={'n':n}
    Y=onehot(ytr[:n])
    m=Ridge(alpha=best['ridge_d1']['alpha']).fit(Str[:n],Y); r['ridge']=float((CLASSES[m.predict(Sva).argmax(1)]==yva).mean())
    m=Ridge(alpha=best['ridge_poly']['alpha']).fit(Ptr[:n],Y); r['ridge_poly']=float((CLASSES[m.predict(Pva).argmax(1)]==yva).mean())
    m=LogisticRegression(C=best['logreg']['C'],max_iter=1000).fit(Str[:n],ytr[:n]); r['logreg']=acc(m,Sva,yva)
    m=SVC(C=best['svm_rbf']['C'],gamma=best['svm_rbf']['gamma']).fit(Str[:n],ytr[:n]); r['svm_rbf']=acc(m,Sva,yva)
    ms=[]
    for s in range(3):
        m=MLPClassifier(hidden_layer_sizes=(64,64),max_iter=150,random_state=s).fit(Str[:n],ytr[:n]); ms.append(acc(m,Sva,yva))
    r['mlp']=float(np.mean(ms)); r['mlp_sd']=float(np.std(ms)); out[str(n)]=r; print(r,flush=True)
save('phaseE',out)
