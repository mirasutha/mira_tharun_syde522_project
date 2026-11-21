import json
R='/home/claude/proj/results/'; F=json.load(open(R+'final.json')); A=json.load(open(R+'phaseA.json')); P=json.load(open(R+'phaseF.json'))
M=F['models']; pct=lambda x:f'{100*x:.1f}'
hp={'OLS linear regression':'--','Ridge regression':r'$\alpha{=}10$','Ridge + polynomial features':r'degree 3, $\alpha{=}1$','Logistic regression':r'L2, $C{=}10$','Perceptron':'10 epochs',
 'SVM, linear kernel':r'$C{=}1$','SVM, polynomial kernel':r'degree 4, $C{=}1$','SVM, RBF kernel':r'$C{=}10$, $\gamma{=}0.1$','MLP, simple':r'1$\times$16','MLP, medium':r'2$\times$64','MLP, deep':r'4$\times$128'}
val={k:M[k]['val'] for k in M}; val['OLS linear regression']=A['ols']['val']
for k in P: val[k]=P[k]['val_mean']
bt=max(m['test_acc'] for m in M.values())
rows=[]
for k,m in M.items():
    t=pct(m['test_acc']); t=r'\textbf{'+t+'}' if m['test_acc']==bt else t
    ft='<0.1' if m['fit_time']<0.05 else f"{m['fit_time']:.1f}"
    rows.append(f"{k} & {hp[k]} & {pct(val[k])} & {t} [{pct(m['ci_lo'])}, {pct(m['ci_hi'])}] & {pct(m['macro_f1'])} & {pct(m['noleak_acc'])} & {ft} \\\\")
tex="\\begin{tabular}{@{}llccccc@{}}\n\\toprule\nModel & Selected setting & Val. & Test acc. [95\\% CI] & Macro-F1 & No-overlap & Fit (s) \\\\\n\\midrule\n"
for i,r in enumerate(rows):
    tex+=r+"\n"
    if i in (1,2,4,7): tex+="\\midrule\n" if i in (2,4,7) else ""
tex+="\\bottomrule\n\\end{tabular}\n"
open('/home/claude/proj/table2.tex','w').write(tex); print(tex)
print({k:round(100*v['test_sd'],2) for k,v in M.items() if v['seeds']>1})
