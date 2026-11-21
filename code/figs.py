import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R='/home/claude/proj/results/'; F='/home/claude/proj/figures/'
L=lambda n: json.load(open(R+n+'.json'))
fams=[f.name for f in font_manager.fontManager.ttflist]
serif='Nimbus Roman' if 'Nimbus Roman' in fams else 'DejaVu Serif'
plt.rcParams.update({'font.family':'serif','font.serif':[serif],'pdf.fonttype':42,'ps.fonttype':42,'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'legend.fontsize':6,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':0.6,'lines.linewidth':1.1,'lines.markersize':3.2,'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.4,'figure.dpi':200})
col={'a':'#0072B2','b':'#D55E00','c':'#009E73','d':'#CC79A7','e':'#E69F00','f':'#56B4E9','k':'#333333'}
A,B,C=L('phaseA'),L('phaseB'),L('phaseC')
def tag(ax,s): ax.text(-0.02,1.04,s,transform=ax.transAxes,fontweight='bold',fontsize=8,va='bottom',ha='right')
# ---- Fig 1: data overview
import pandas as pd
df=pd.read_csv('/mnt/user-data/uploads/letter-recognition.csv'); df.columns=[c.strip() for c in df.columns]
fig,ax=plt.subplots(1,2,figsize=(5.5,1.9),gridspec_kw={'width_ratios':[1,1.15]})
vc=df['letter'].value_counts().sort_index(); ax[0].bar(range(26),vc.values,color=col['a'],width=0.75); ax[0].set_xticks(range(26)); ax[0].set_xticklabels(vc.index,fontsize=5.5); ax[0].set_ylim(0,900); ax[0].set_ylabel('examples'); ax[0].set_xlabel('letter'); ax[0].set_title('Class counts (20,000 total)'); tag(ax[0],'(a)')
ax[1].boxplot([df[c].values for c in df.columns[1:]],widths=0.6,flierprops={'markersize':1.5},medianprops={'color':col['b']},boxprops={'linewidth':0.6},whiskerprops={'linewidth':0.6},capprops={'linewidth':0.6}); ax[1].set_xticklabels(df.columns[1:],rotation=60,fontsize=5.5,ha='right'); ax[1].set_ylabel('feature value (0–15)'); ax[1].set_title('Distribution of the 16 features'); tag(ax[1],'(b)')
fig.tight_layout(pad=0.4); fig.savefig(F+'fig1_data.pdf'); plt.close(fig)
# ---- Fig 2: simple models
fig,ax=plt.subplots(1,3,figsize=(5.5,1.85))
for d,c,m in [(1,col['a'],'o'),(2,col['b'],'s'),(3,col['c'],'^')]:
    r=[x for x in A['ridge'] if x['degree']==d]; ax[0].plot([x['alpha'] for x in r],[x['val'] for x in r],color=c,marker=m,label=f'degree {d}')
ax[0].set_xscale('log'); ax[0].set_xlabel(r'ridge penalty $\alpha$'); ax[0].set_ylabel('validation accuracy'); ax[0].set_title('Ridge (one-hot targets)'); ax[0].legend(loc='lower left'); ax[0].set_ylim(0.35,1.0); tag(ax[0],'(a)')
for pen,c,m in [('l2',col['a'],'o'),('l1',col['b'],'s')]:
    r=[x for x in A['logreg'] if x['penalty']==pen]; ax[1].plot([x['C'] for x in r],[x['val'] for x in r],color=c,marker=m,label=pen.upper())
ax[1].set_xscale('log'); ax[1].set_xlabel('inverse penalty $C$'); ax[1].set_title('Logistic regression'); ax[1].legend(loc='center right'); ax[1].set_ylim(0,1.0); tag(ax[1],'(b)')
ep=[x['epochs'] for x in A['perceptron']]; mu=np.array([x['val_mean'] for x in A['perceptron']]); sd=np.array([x['val_sd'] for x in A['perceptron']])
ax[2].errorbar(ep,mu,yerr=sd,color=col['d'],marker='o',capsize=2,label='val (mean ± sd, 5 seeds)'); ax[2].plot(ep,[x['train_mean'] for x in A['perceptron']],color=col['k'],ls='--',label='train'); ax[2].set_xscale('log'); ax[2].set_xlabel('training epochs'); ax[2].set_title('Perceptron'); ax[2].set_ylim(0.4,0.65); ax[2].legend(loc='lower right'); tag(ax[2],'(c)')
fig.tight_layout(pad=0.4); fig.savefig(F+'fig2_simple.pdf'); plt.close(fig)
# ---- Fig 3: SVM
Cs=sorted({x['C'] for x in B['rbf']}); gs=sorted({x['gamma'] for x in B['rbf']})
M=np.array([[next(x['val'] for x in B['rbf'] if x['C']==c and x['gamma']==g) for g in gs] for c in Cs])
fig,ax=plt.subplots(1,2,figsize=(5.5,2.15),gridspec_kw={'width_ratios':[1.25,1]})
im=ax[0].imshow(M,origin='lower',cmap='viridis',vmin=0.3,vmax=1.0,aspect='auto'); ax[0].set_xticks(range(len(gs))); ax[0].set_xticklabels([f'{g:g}' for g in gs]); ax[0].set_yticks(range(len(Cs))); ax[0].set_yticklabels([f'{c:g}' for c in Cs]); ax[0].set_xlabel(r'kernel width $\gamma$'); ax[0].set_ylabel('$C$'); ax[0].set_title('RBF SVM validation accuracy'); ax[0].grid(False)
for i in range(len(Cs)):
    for j in range(len(gs)): ax[0].text(j,i,f'{M[i,j]*100:.1f}',ha='center',va='center',fontsize=5.5,color='white' if M[i,j]<0.8 else 'black')
bi=np.unravel_index(M.argmax(),M.shape); ax[0].add_patch(plt.Rectangle((bi[1]-0.5,bi[0]-0.5),1,1,fill=False,ec=col['b'],lw=1.4)); tag(ax[0],'(a)')
ax[1].plot([x['C'] for x in B['linear']],[x['val'] for x in B['linear']],color=col['a'],marker='o',label='linear')
for d,c,m in [(2,col['b'],'s'),(3,col['c'],'^'),(4,col['d'],'v')]:
    r=[x for x in B['poly'] if x['degree']==d]; ax[1].plot([x['C'] for x in r],[x['val'] for x in r],color=c,marker=m,label=f'poly, degree {d}')
ax[1].plot(Cs,[max(x['val'] for x in B['rbf'] if x['C']==c) for c in Cs],color=col['k'],marker='D',label=r'RBF (best $\gamma$)'); ax[1].set_xscale('log'); ax[1].set_xlabel('$C$'); ax[1].set_ylabel('validation accuracy'); ax[1].set_title('Kernel comparison'); ax[1].set_ylim(0.6,1.0); ax[1].legend(loc='lower right'); tag(ax[1],'(b)')
fig.tight_layout(pad=0.4); fig.savefig(F+'fig3_svm.pdf'); plt.close(fig)
# ---- Fig 4: MLP
fig,ax=plt.subplots(2,2,figsize=(5.5,3.5))
ws=[16,32,64,128]
for d,c,m in [(1,col['a'],'o'),(2,col['b'],'s'),(3,col['c'],'^'),(4,col['d'],'v')]:
    ax[0,0].plot(ws,[next(x['val'] for x in C['grid'] if x['depth']==d and x['width']==w) for w in ws],color=c,marker=m,label=f'{d} hidden layer'+('s' if d>1 else ''))
ax[0,0].set_xscale('log',base=2); ax[0,0].set_xticks(ws); ax[0,0].set_xticklabels(ws); ax[0,0].set_xlabel('units per hidden layer'); ax[0,0].set_ylabel('validation accuracy'); ax[0,0].set_title('Depth and width'); ax[0,0].legend(loc='lower right'); tag(ax[0,0],'(a)')
ax[0,1].plot([x['lr'] for x in C['lr']],[x['val'] for x in C['lr']],color=col['a'],marker='o',label='validation'); ax[0,1].plot([x['lr'] for x in C['lr']],[x['train'] for x in C['lr']],color=col['k'],ls='--',marker='.',label='train'); ax[0,1].set_xscale('log'); ax[0,1].set_xlabel('learning rate'); ax[0,1].set_title('Learning rate (64-64 net)'); ax[0,1].legend(loc='lower center'); tag(ax[0,1],'(b)')
al=[x for x in C['alpha']]; xs=[max(x['alpha'],1e-6) for x in al]
ax[1,0].plot(xs,[x['val'] for x in al],color=col['a'],marker='o',label='validation'); ax[1,0].plot(xs,[x['train'] for x in al],color=col['k'],ls='--',marker='.',label='train'); ax[1,0].set_xscale('log'); ax[1,0].set_xlabel(r'L2 penalty $\alpha$ (0 plotted at $10^{-6}$)'); ax[1,0].set_ylabel('accuracy'); ax[1,0].set_title('Weight decay (64-64 net)'); ax[1,0].legend(loc='lower left'); tag(ax[1,0],'(c)')
for nm,c,lab in [('simple',col['a'],'16'),('medium',col['b'],'64-64'),('deep',col['c'],'4×128')]:
    cv=C['curves'][nm]; e=np.arange(1,len(cv['val'])+1); ax[1,1].plot(e,cv['val'],color=c,label=f'{lab} val'); ax[1,1].plot(e,cv['train'],color=c,ls=':',label=f'{lab} train')
ax[1,1].set_xlabel('epoch'); ax[1,1].set_title('Training curves'); ax[1,1].legend(loc='lower right',ncol=2,columnspacing=0.8); ax[1,1].set_ylim(0.5,1.01); tag(ax[1,1],'(d)')
fig.tight_layout(pad=0.4,h_pad=0.8); fig.savefig(F+'fig4_mlp.pdf'); plt.close(fig)
# ---- Fig 5: training size + final comparison
import os
if os.path.exists(R+'phaseE.json') and os.path.exists(R+'final.json'):
    E=L('phaseE'); Fi=L('final'); ns=[int(k) for k in E]
    fig,ax=plt.subplots(1,2,figsize=(5.5,2.3),gridspec_kw={'width_ratios':[1,1.15]})
    for k,c,m,lab in [('ridge',col['a'],'o','ridge (linear)'),('logreg',col['e'],'x','logistic reg.'),('ridge_poly',col['d'],'v','ridge + poly'),('mlp',col['c'],'^','MLP 64-64'),('svm_rbf',col['b'],'s','RBF SVM')]:
        ax[0].plot(ns,[E[str(n)][k] for n in ns],color=c,marker=m,label=lab)
    ax[0].set_xscale('log'); ax[0].set_xlabel('training examples'); ax[0].set_ylabel('validation accuracy'); ax[0].set_title('Effect of training-set size'); ax[0].legend(loc='upper left'); tag(ax[0],'(a)')
    ms=Fi['models']; names=list(ms)[::-1]; y=np.arange(len(names))
    ax[1].errorbar([ms[n]['test_acc'] for n in names],y,xerr=[[ms[n]['test_acc']-ms[n]['ci_lo'] for n in names],[ms[n]['ci_hi']-ms[n]['test_acc'] for n in names]],fmt='o',color=col['b'],ecolor=col['k'],capsize=2,lw=0.8)
    ax[1].set_yticks(y); ax[1].set_yticklabels(names,fontsize=6); ax[1].set_xlabel('test accuracy (95% bootstrap CI)'); ax[1].set_title('Final comparison on the test set'); ax[1].set_xlim(0.45,1.0); tag(ax[1],'(b)')
    fig.tight_layout(pad=0.4); fig.savefig(F+'fig5_final.pdf'); plt.close(fig)
    # ---- Fig 6: confusion matrix (errors only)
    cm=np.array(Fi['confusion_rbf']); off=cm.copy(); np.fill_diagonal(off,0)
    fig,ax=plt.subplots(figsize=(3.0,2.8)); im=ax.imshow(off,cmap='Reds',vmin=0); ax.set_xticks(range(26)); ax.set_yticks(range(26)); lab=[chr(65+i) for i in range(26)]; ax.set_xticklabels(lab,fontsize=5); ax.set_yticklabels(lab,fontsize=5); ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.grid(False); ax.set_title('RBF SVM errors on the test set'); fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03).ax.tick_params(labelsize=5.5)
    fig.tight_layout(pad=0.4); fig.savefig(F+'fig6_confusion.pdf'); plt.close(fig)
print('figs ok')
