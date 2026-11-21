# validation accuracy of the 3 MLP architectures under the FINAL (tuned) lr/alpha, 5 seeds, train-sub -> val
from common import *
from sklearn.neural_network import MLPClassifier
F=load('final'); LR,AL=F['mlp_lr'],F['mlp_alpha']; out={}
for nm,h in [('MLP, simple',(16,)),('MLP, medium',(64,64)),('MLP, deep',(128,)*4)]:
    v=[acc(MLPClassifier(hidden_layer_sizes=h,learning_rate_init=LR,alpha=AL,max_iter=150,random_state=s).fit(Str,ytr),Sva,yva) for s in range(5)]
    out[nm]={'val_mean':float(np.mean(v)),'val_sd':float(np.std(v))}; print(nm,round(np.mean(v),4),round(np.std(v),4),flush=True)
save('phaseF',out)
