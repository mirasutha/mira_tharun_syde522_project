from common import *
from sklearn.neural_network import MLPClassifier
MAXIT=150
def mk(h,seed=0,lr=1e-3,alpha=1e-4,bs=200): return MLPClassifier(hidden_layer_sizes=h,learning_rate_init=lr,alpha=alpha,batch_size=bs,max_iter=MAXIT,random_state=seed)
def run(h,**kw):
    t=time.time(); m=mk(h,**kw).fit(Str,ytr)
    return {'val':acc(m,Sva,yva),'train':acc(m,Str,ytr),'iters':int(m.n_iter_),'hit_max':bool(m.n_iter_>=MAXIT),'time':time.time()-t}
out={'grid':[],'lr':[],'alpha':[],'curves':{}}
for d in [1,2,3,4]:
    for w in [16,32,64,128]:
        r=run((w,)*d); r.update(depth=d,width=w); out['grid'].append(r); print('grid',d,w,{k:round(v,3) if isinstance(v,float) else v for k,v in r.items()},flush=True)
save('phaseC',out)
for lr in [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]:
    r=run((64,64),lr=lr); r['lr']=lr; out['lr'].append(r); print('lr',lr,round(r['val'],3),flush=True)
for a in [0,1e-5,1e-4,1e-3,1e-2,1e-1,1]:
    r=run((64,64),alpha=a); r['alpha']=a; out['alpha'].append(r); print('alpha',a,round(r['val'],3),flush=True)
save('phaseC',out)
for name,h in [('simple',(16,)),('medium',(64,64)),('deep',(128,128,128,128))]:
    m=MLPClassifier(hidden_layer_sizes=h,learning_rate_init=1e-3,alpha=1e-4,batch_size=200,random_state=0)
    tr,va,loss=[],[],[]
    for ep in range(100):
        m.partial_fit(Str,ytr,classes=CLASSES); tr.append(acc(m,Str,ytr)); va.append(acc(m,Sva,yva)); loss.append(float(m.loss_))
    out['curves'][name]={'train':tr,'val':va,'loss':loss}; print('curve',name,round(tr[-1],3),round(va[-1],3),flush=True)
save('phaseC',out)
