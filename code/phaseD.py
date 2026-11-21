# FINAL: retrain chosen configs on full 16k pool, evaluate on the 4k test set ONCE
from common import *
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix
best=load('best'); C=load('phaseC')
Sp,St=scaled(Xpool,Xtest)
cidx={c:i for i,c in enumerate(CLASSES)}; yt=np.array([cidx[c] for c in ytest]); Yp=onehot(ypool)
# MLP lr/alpha: confirm the coordinate-wise choice on validation (medium net), else keep defaults
lr_s,al_s=best['mlp_lr']['lr'],best['mlp_alpha']['alpha']
def valmlp(lr,al):
    m=MLPClassifier(hidden_layer_sizes=(64,64),learning_rate_init=lr,alpha=al,max_iter=150,random_state=0).fit(Str,ytr); return acc(m,Sva,yva)
v_sel,v_def=valmlp(lr_s,al_s),valmlp(1e-3,1e-4); LR,AL=(lr_s,al_s) if v_sel>=v_def else (1e-3,1e-4)
print('mlp lr/alpha: selected',lr_s,al_s,round(v_sel,4),'default',round(v_def,4),'->',LR,AL,flush=True)
def ls(alpha,deg,ols=False):
    def f(seed):
        A,B=Sp,St
        if deg>1:
            P=PolynomialFeatures(deg,include_bias=False); A=P.fit_transform(Sp); B=P.transform(St); sc=StandardScaler().fit(A); A,B=sc.transform(A),sc.transform(B)
        t=time.time(); m=LinearRegression() if ols else Ridge(alpha=alpha); m.fit(A,Yp); dt=time.time()-t
        return CLASSES[m.predict(B).argmax(1)],float((CLASSES[m.predict(A).argmax(1)]==ypool).mean()),dt,None
    return f
def sk(mk):
    def f(seed):
        m=mk(seed); t=time.time(); m.fit(Sp,ypool); dt=time.time()-t
        return m.predict(St),acc(m,Sp,ypool),dt,(int(getattr(m,'n_iter_',0)) if hasattr(m,'n_iter_') and np.isscalar(m.n_iter_) else None)
    return f
b=best
models=[
 ('OLS linear regression',{},1,ls(0,1,True),None),
 ('Ridge regression',{'alpha':b['ridge_d1']['alpha']},1,ls(b['ridge_d1']['alpha'],1),b['ridge_d1']['val']),
 ('Ridge + polynomial features',{'degree':b['ridge_poly']['degree'],'alpha':b['ridge_poly']['alpha']},1,ls(b['ridge_poly']['alpha'],int(b['ridge_poly']['degree'])),b['ridge_poly']['val']),
 ('Logistic regression',{'penalty':b['logreg']['penalty'],'C':b['logreg']['C']},1,sk(lambda s:LogisticRegression(C=b['logreg']['C'],max_iter=1000) if b['logreg']['penalty']=='l2' else LogisticRegression(C=b['logreg']['C'],penalty='l1',solver='saga',max_iter=200,tol=1e-3)),b['logreg']['val']),
 ('Perceptron',{'epochs':b['perceptron']['epochs']},5,sk(lambda s:Perceptron(max_iter=int(b['perceptron']['epochs']),tol=None,random_state=s)),b['perceptron']['val_mean']),
 ('SVM, linear kernel',{'C':b['svm_linear']['C']},1,sk(lambda s:SVC(kernel='linear',C=b['svm_linear']['C'],max_iter=2000000)),b['svm_linear']['val']),
 ('SVM, polynomial kernel',{'degree':b['svm_poly']['degree'],'C':b['svm_poly']['C']},1,sk(lambda s:SVC(kernel='poly',degree=int(b['svm_poly']['degree']),coef0=1,C=b['svm_poly']['C'])),b['svm_poly']['val']),
 ('SVM, RBF kernel',{'C':b['svm_rbf']['C'],'gamma':b['svm_rbf']['gamma']},1,sk(lambda s:SVC(C=b['svm_rbf']['C'],gamma=b['svm_rbf']['gamma'])),b['svm_rbf']['val']),
]
grid={(r['depth'],r['width']):r for r in C['grid']}
for nm,h,key in [('MLP, simple',(16,),(1,16)),('MLP, medium',(64,64),(2,64)),('MLP, deep',(128,)*4,(4,128))]:
    models.append((nm,{'hidden':list(h),'lr':LR,'alpha':AL},5,sk(lambda s,h=h:MLPClassifier(hidden_layer_sizes=h,learning_rate_init=LR,alpha=AL,max_iter=150,random_state=s)),grid[key]['val']))
BIDX=np.random.RandomState(123).randint(0,4000,(1000,4000))
final={'mlp_lr':LR,'mlp_alpha':AL,'models':{}}; CORR={}; PRED={}; BOOT={}
for name,hp,nseeds,fn,vacc in models:
    t0=time.time(); res=[fn(s) for s in range(nseeds)]
    P=np.array([[cidx[c] for c in r[0]] for r in res]); corr=(P==yt[None,:]); accs=corr.mean(1)
    boots=np.array([corr[:,BIDX[b_]].mean() for b_ in range(1000)]); ci=np.percentile(boots,[2.5,97.5])
    final['models'][name]={'hyper':hp,'val':vacc,'seeds':nseeds,'test_acc':float(accs.mean()),'test_sd':float(accs.std()),'ci_lo':float(ci[0]),'ci_hi':float(ci[1]),
      'macro_f1':float(np.mean([f1_score(yt,p,average='macro') for p in P])),'noleak_acc':float(corr[:,NOLEAK].mean()),'train_acc':float(np.mean([r[1] for r in res])),
      'fit_time':float(np.mean([r[2] for r in res])),'n_iter':[r[3] for r in res]}
    CORR[name]=corr; PRED[name]=P; BOOT[name]=boots
    print(name,hp,'test %.4f [%.4f,%.4f] f1 %.4f noleak %.4f time %.1fs (%.0fs total)'%(accs.mean(),ci[0],ci[1],final['models'][name]['macro_f1'],final['models'][name]['noleak_acc'],final['models'][name]['fit_time'],time.time()-t0),flush=True)
    save('final',final)
np.savez_compressed(f'{ROOT}/results/final_arrays.npz',**{('corr_'+k):v for k,v in CORR.items()},**{('pred_'+k):v for k,v in PRED.items()},**{('boot_'+k):v for k,v in BOOT.items()})
# paired bootstrap differences + confusion matrix for the RBF SVM
names=list(CORR); d={}
def pair(a,b_): x=BOOT[a]-BOOT[b_]; return [float((CORR[a].mean()-CORR[b_].mean())),float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]
mlps=['MLP, simple','MLP, medium','MLP, deep']; bm=max(mlps,key=lambda k:final['models'][k]['val'])
final['best_mlp_by_val']=bm
final['pairs']={'rbf_vs_bestmlp':pair('SVM, RBF kernel',bm),'rbf_vs_polyridge':pair('SVM, RBF kernel','Ridge + polynomial features'),'rbf_vs_logreg':pair('SVM, RBF kernel','Logistic regression'),'bestmlp_vs_logreg':pair(bm,'Logistic regression'),'medium_vs_deep':pair('MLP, medium','MLP, deep'),'simple_vs_medium':pair('MLP, medium','MLP, simple'),'rbf_vs_polysvm':pair('SVM, RBF kernel','SVM, polynomial kernel'),'ridge_vs_ols':pair('Ridge regression','OLS linear regression')}
cm=confusion_matrix(yt,PRED['SVM, RBF kernel'][0],labels=range(26)); final['confusion_rbf']=cm.tolist()
off=[(int(cm[i,j]),CLASSES[i],CLASSES[j]) for i in range(26) for j in range(26) if i!=j]; off.sort(reverse=True); final['top_confusions']=[[a,b_,c] for a,b_,c in off[:8]]
save('final',final); print(final['pairs']); print(final['top_confusions'])
