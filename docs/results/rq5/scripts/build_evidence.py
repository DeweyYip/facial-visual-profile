"""Recompute and audit archived RQ5 summaries; export publication figures.

Run: python scripts/build_evidence.py (from any directory).
Requires numpy, pandas, matplotlib, reportlab. No torch or model inference.
"""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'raw'; TABLES=ROOT/'tables'; FIG=ROOT/'figures'
for p in (TABLES,FIG):p.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,label,tol=1e-12):
    assert np.allclose(a,b,rtol=0,atol=tol),label
MODELS=['A','N0','N1','MTL']; SEEDS=[0,1,2]
FAMILIES=['blur','brightness','rotation','occlusion','jpeg']
LABELS=['clean']+[f'{f}_s{s}' for f in FAMILIES for s in range(1,5)]
METRICS=['mAP','frozen_F1','F1_at_05']; TITLES={'mAP':'mAP','frozen_F1':'Frozen-threshold macro-F1','F1_at_05':'Macro-F1 at 0.5'}
COLORS={'A':'#777777','N0':'#0072B2','N1':'#D55E00','MTL':'#009E73'}
complete=read(RAW/'rq5_complete.json'); frozen=read(RAW/'frozen_config.json')
source=read(RAW/'notebook_execution_binding.json'); baseline_binding=read(RAW/'rq4_baseline_binding.json')
conditions=pd.read_csv(RAW/'condition_results.csv'); attrs=pd.read_csv(RAW/'attribute_results.csv')
assert conditions.shape==(252,7) and attrs.shape==(6048,6)
assert not conditions.duplicated(['model','seed','condition']).any()
assert not attrs.duplicated(['model','seed','condition','attribute']).any()
assert set(conditions.model)==set(MODELS) and set(conditions.seed)==set(SEEDS)
attribute_order=list(attrs[(attrs.model=='N0')&(attrs.seed==0)&(attrs.condition=='clean')].attribute)
assert len(attribute_order)==24
for key,g in conditions.groupby(['model','seed']):assert set(g.condition)==set(LABELS),key
assert np.isfinite(conditions[METRICS]).all().all()
assert ((conditions[METRICS]>=0)&(conditions[METRICS]<=1)).all().all()
for row in read(ROOT/'source_manifest.json'):
    assert sha(ROOT/row['path'])==row['sha256'],row['path']
training=[];histories={};nme=[];raw_checks=0
for model in MODELS:
 for seed in SEEDS:
    key=f'{model}_seed{seed}'
    folder=RAW/('baseline' if model=='A' else 'runs')/key
    metrics=read(folder/'test_metrics.json'); cfg=read(folder/'config.json')
    done=read(folder/'train_complete.json'); best=read(folder/'best.json');hist=read(folder/'history.json')
    assert set(metrics)==set(LABELS)
    assert [h['epoch'] for h in hist]==list(range(1,done['epochs']+1))
    chosen=max(hist,key=lambda h:h['clean_validation_mAP'])
    assert chosen['epoch']==best['epoch']==done['best']['epoch']
    assert done['epochs']>=4 and done['epochs']-best['epoch']>=2 # min-epoch gate can yield >2 stale epochs
    for end in range(1,len(hist)):
        prior=hist[:end];prior_best=max(prior,key=lambda h:h['clean_validation_mAP'])
        if model!='A':assert not (prior[-1]['epoch']>=4 and prior[-1]['epoch']-prior_best['epoch']>=2), 'Missed earlier stop'
    threshold=read(folder/'clean_val_thresholds.json')
    assert threshold['attribute_order']==attribute_order
    assert len(threshold['thresholds'])==24 and all(.05<=v<=.95 for v in threshold['thresholds'])
    assert read(folder/'test_complete.json')['conditions']==21
    if model!='A':
        binding=read(folder/'test_binding.json'); recorded=complete['runs'][key]
        assert sha(folder/'test_metrics.json')==recorded['test_sha256']
        assert sha(folder/'clean_val_thresholds.json')==recorded['thresholds_sha256']==binding['thresholds']
        assert threshold['checkpoint_sha256']==recorded['checkpoint_sha256']==binding['checkpoint']
        assert binding['sources']=={k:v for k,v in source['sources'].items() if k in binding['sources']}
        assert binding['evaluation_script']==source['sources']['scripts/evaluate_rq5.py']
        assert cfg['config']==frozen and cfg['attributes']==attribute_order
        assert cfg['seed']==seed and cfg['model']==model
        assert recorded['training']==done
    else:
        for name,expected in baseline_binding[str(seed)].items():
            if name!='best_model.pt':assert sha(folder/name)==expected,(key,name)
    for label,m in metrics.items():
        assert m['n']==19962
        assert len(m['AP_by_attr'])==len(m['F1_by_attr'])==24
        close(np.mean(m['AP_by_attr']),m['mAP'],(key,label,'macro AP'))
        close(np.mean(m['F1_by_attr']),m['macro_f1'],(key,label,'macro F1'))
        row=conditions[(conditions.model==model)&(conditions.seed==seed)&(conditions.condition==label)].iloc[0]
        close(row[METRICS].to_numpy(dtype=float),[m['mAP'],m['macro_f1'],m['F1_at_05']],(key,label))
        a=attrs[(attrs.model==model)&(attrs.seed==seed)&(attrs.condition==label)].set_index('attribute').loc[attribute_order]
        close(a.AP,m['AP_by_attr'],(key,label,'attribute AP'))
        close(a.F1,m['F1_by_attr'],(key,label,'attribute F1'))
        raw_checks+=1
        if model=='MTL':
            diagnostic=m['landmark_NME'];assert diagnostic['count']==19962
            nme.append(dict(model=model,seed=seed,condition=label,family=row.family,
                mean_NME_pct=100*diagnostic['mean'],median_NME_pct=100*diagnostic['median'],n=19962))
    histories[(model,seed)]=hist
    training.append(dict(model=model,seed=seed,epochs=done['epochs'],best_epoch=best['epoch'],
        best_validation_mAP=chosen['clean_validation_mAP'],stop_reason='early_stopping',
        params={'A':11188824,'N0':11249368,'N1':11250776,'MTL':11193954}[model]))
# Do not infer baseline parameter count from an unaudited value.
training=pd.DataFrame(training);training.loc[training.model=='A','params']=np.nan
training.to_csv(TABLES/'training_summary.csv',index=False)
pd.DataFrame(nme).to_csv(TABLES/'mtl_landmark_diagnostics.csv',index=False)
scale=read(RAW/'lambda_scale_check.json')
mean_bce=np.mean([b['BCE'] for b in scale['batches']]);mean_mse=np.mean([b['MSE'] for b in scale['batches']])
assert len(scale['batches'])==4
assert float(f'{mean_bce/mean_mse:.6g}')==frozen['mtl_lambda']==complete['lambda_value']==1.89811

# The experimental unit for uncertainty summaries is a seed, never a condition/image.
seed_rows=[]
for model in MODELS:
 for seed in SEEDS:
    df=conditions[(conditions.model==model)&(conditions.seed==seed)]
    for scope in ['clean','corrupted']:
        subset=df[df.condition=='clean'] if scope=='clean' else df[df.condition!='clean']
        assert len(subset)==(1 if scope=='clean' else 20)
        seed_rows.append(dict(model=model,seed=seed,scope=scope,**subset[METRICS].mean().to_dict()))
seed_table=pd.DataFrame(seed_rows);seed_table.to_csv(TABLES/'scores_by_seed.csv',index=False)
summary=seed_table.groupby(['scope','model'])[METRICS].agg(['mean','std']).reindex(pd.MultiIndex.from_product([['clean','corrupted'],MODELS],names=['scope','model']))
summary.to_csv(TABLES/'scores_mean_sample_sd.csv')
old=pd.read_csv(RAW/'corruption_summary.csv',header=[0,1],index_col=0)
for model in MODELS:
 for metric in METRICS:
    for stat in ['mean','std']:close(summary.loc[('corrupted',model),(metric,stat)],old.loc[model,(metric,stat)],'archived summary')
CONTRASTS=[('N1','N0'),('MTL','A'),('N0','A')]
paired=[]
for scope in ['clean','corrupted']:
    idx=seed_table[seed_table.scope==scope].set_index(['model','seed'])
    for a,b in CONTRASTS:
        for seed in SEEDS:
            paired.append(dict(comparison=f'{a}-{b}',scope=scope,seed=seed,**{f'delta_{m}_pp':100*(idx.loc[(a,seed),m]-idx.loc[(b,seed),m]) for m in METRICS}))
paired=pd.DataFrame(paired)
archive=pd.read_csv(RAW/'paired_seed_contrasts.csv').set_index(['comparison','scope','seed']).sort_index()
current=paired.set_index(['comparison','scope','seed']).sort_index()
close(archive[current.columns],current,'paired CSV')
paired.to_csv(TABLES/'paired_seed_contrasts.csv',index=False)
delta_cols=[f'delta_{m}_pp' for m in METRICS]
paired_summary=paired.groupby(['comparison','scope'])[delta_cols].agg(['mean','std'])
paired_summary.to_csv(TABLES/'paired_mean_sample_sd.csv')
family=conditions[conditions.condition!='clean'].groupby(['model','seed','family'])[METRICS].mean().reset_index()
family.to_csv(TABLES/'family_scores_by_seed.csv',index=False)
conditions.groupby(['model','condition'])[METRICS].agg(['mean','std']).to_csv(TABLES/'condition_mean_sample_sd.csv')
attrs.groupby(['model','condition','attribute'])[['AP','F1']].agg(['mean','std']).to_csv(TABLES/'attribute_mean_sample_sd.csv')
pd.DataFrame(nme).groupby('condition')[['mean_NME_pct']].agg(['mean','std']).to_csv(TABLES/'mtl_NME_condition_mean_sample_sd.csv')
family_deltas=[]
for f in FAMILIES:
 idx=family[family.family==f].set_index(['model','seed'])
 for a,b in CONTRASTS:
  for seed in SEEDS:
   family_deltas.append(dict(comparison=f'{a}-{b}',family=f,seed=seed,**{f'delta_{m}_pp':100*(idx.loc[(a,seed),m]-idx.loc[(b,seed),m]) for m in METRICS}))
family_deltas=pd.DataFrame(family_deltas);family_deltas.to_csv(TABLES/'family_paired_differences.csv',index=False)
attr_corrupt=attrs[attrs.condition!='clean'].groupby(['model','seed','attribute'])[['AP','F1']].mean().reset_index()
attr_deltas=[]
for a,b in CONTRASTS:
 idx=attr_corrupt.set_index(['model','seed','attribute'])
 for seed in SEEDS:
  for attribute in attribute_order:
   attr_deltas.append(dict(comparison=f'{a}-{b}',seed=seed,attribute=attribute,
      delta_AP_pp=100*(idx.loc[(a,seed,attribute),'AP']-idx.loc[(b,seed,attribute),'AP']),
      delta_F1_pp=100*(idx.loc[(a,seed,attribute),'F1']-idx.loc[(b,seed,attribute),'F1'])))
attr_deltas=pd.DataFrame(attr_deltas);attr_deltas.to_csv(TABLES/'attribute_paired_differences.csv',index=False)
for label,frame,keys,cols in [('family',family_deltas,['comparison','family'],delta_cols),('attribute',attr_deltas,['comparison','attribute'],['delta_AP_pp','delta_F1_pp'])]:
 rows=[]
 for key,g in frame.groupby(keys):
  for metric in cols:
   v=g[metric].to_numpy()
   rows.append(dict(zip(keys,key),metric=metric,mean_pp=v.mean(),sample_sd_pp=v.std(ddof=1),positive_seeds=int((v>0).sum()),negative_seeds=int((v<0).sum()),zero_seeds=int((v==0).sum())))
 pd.DataFrame(rows).to_csv(TABLES/f'{label}_contrast_summary.csv',index=False)

# Export compact paper table in percent; differences are percentage points.
lines=['| Scope | Model | mAP (%) | Frozen macro-F1 (%) | F1 at 0.5 (%) |','|---|---|---:|---:|---:|']
tex=['% Mean and sample SD over seeds 0,1,2; percent units.','\\begin{tabular}{llccc}','\\toprule','Scope & Model & mAP & Frozen macro-F1 & F1@0.5 \\\\','\\midrule']
for (scope,model),row in summary.iterrows():
 vals=[f"{100*row[(m,'mean')]:.3f} ± {100*row[(m,'std')]:.3f}" for m in METRICS]
 lines.append('| '+' | '.join([scope,model]+vals)+' |')
 tv=[f"${100*row[(m,'mean')]:.3f} \\pm {100*row[(m,'std')]:.3f}$" for m in METRICS]
 tex.append(' & '.join([scope,model]+tv)+' \\\\')
tex+=['\\bottomrule','\\end{tabular}']
(TABLES/'paper_main_table.md').write_text('\n'.join(lines)+'\n')
(TABLES/'paper_main_table.tex').write_text('\n'.join(tex)+'\n')

# Restrained scientific styling; all labels/captions in English for the manuscript.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white','figure.facecolor':'white'})
CAPTIONS={}
def export(fig,name,caption):
 for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig);CAPTIONS[name]=caption
seed_markers=['o','s','^'];jitter=[-.09,0,.09]

fig,axes=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
for i,scope in enumerate(['clean','corrupted']):
 for j,metric in enumerate(['mAP','frozen_F1']):
  ax=axes[i,j];ax.axhline(0,color='.55',lw=.8,zorder=0)
  for x,(a,b) in enumerate(CONTRASTS):
   v=paired[(paired.scope==scope)&(paired.comparison==f'{a}-{b}')].sort_values('seed')[f'delta_{metric}_pp'].to_numpy()
   for seed,y in enumerate(v):ax.scatter(x+jitter[seed],y,s=30,marker=seed_markers[seed],color=COLORS[a],edgecolors='white',linewidth=.4,zorder=3,label=f'Seed {seed}' if x==0 else None)
   ax.errorbar(x+.22,v.mean(),yerr=v.std(ddof=1),fmt='D',color='black',ms=4,capsize=3,lw=1,zorder=4)
  ax.set_xticks(range(3),['N1 − N0','MTL − A','N0 − A']);ax.set_xlim(-.4,2.55)
  ax.set_ylabel('Difference (percentage points)')
  ax.set_title(f"({chr(97+2*i+j)}) {scope.capitalize()}: {TITLES[metric]}",loc='left')
  ax.grid(axis='y',color='.92',lw=.6);ax.set_axisbelow(True)
from matplotlib.lines import Line2D
handles=[Line2D([],[],ls='',marker=seed_markers[k],color='.35',label=f'Seed {k}') for k in SEEDS]
handles.append(Line2D([],[],ls='',marker='D',color='black',label='Mean ± sample SD'))
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.05),ncol=4,frameon=False,fontsize=8)
export(fig,'fig01_paired_effects','Paired effects for clean and corrupted test inputs. Colored symbols show seeds 0/1/2; black diamonds show the mean and error bars the sample SD of the three paired differences. Corrupted scores first average the 20 non-clean conditions within each seed. Error bars are not confidence intervals; no significance test is implied. N1-N0 and MTL-A are primary contrasts; N0-A is explanatory.')

fig,axes=plt.subplots(2,5,figsize=(10,4.6),sharey='row',layout='constrained')
family_names={'blur':'Gaussian blur','brightness':'Darkening','rotation':'Rotation','occlusion':'Lower-face occlusion','jpeg':'JPEG compression'}
curve_rows=[]
for j,f in enumerate(FAMILIES):
 labels=['clean']+[f'{f}_s{s}' for s in range(1,5)]
 for i,metric in enumerate(['mAP','frozen_F1']):
  ax=axes[i,j]
  for model in MODELS:
   df=conditions[(conditions.model==model)&conditions.condition.isin(labels)]
   g=df.groupby('condition')[metric]
   means=g.mean().reindex(labels).to_numpy()*100;sd=g.std().reindex(labels).to_numpy()*100
   ax.plot(range(5),means,'o-',color=COLORS[model],ms=3,label=model)
   ax.fill_between(range(5),means-sd,means+sd,color=COLORS[model],alpha=.10,lw=0)
   for sev,mean,std in zip(range(5),means,sd):curve_rows.append(dict(family=f,metric=metric,model=model,severity=sev,mean_pct=mean,sample_sd_pct=std))
  ax.set_xticks(range(5),['C','1','2','3','4']);ax.grid(axis='y',color='.92',lw=.6)
  if i==0:ax.set_title(family_names[f])
  if j==0:ax.set_ylabel(TITLES[metric]+' (%)')
  if i==1:ax.set_xlabel('Severity')
axes[0,0].legend(frameon=False,ncol=2,fontsize=7)
export(fig,'fig02_corruption_curves','Test scores versus corruption severity. C denotes clean; levels 1-4 follow the frozen benchmark. Lines are means over three seeds and bands are plus/minus one sample SD, not confidence intervals. Each corruption is applied independently to the clean canvas. The clean point is repeated for reference in each panel and excluded from corrupted aggregates.')
pd.DataFrame(curve_rows).to_csv(TABLES/'curve_plot_data.csv',index=False)

fig,axes=plt.subplots(2,2,figsize=(8.0,5.8),layout='constrained')
for i,comparison in enumerate(['N1-N0','MTL-A']):
 for j,metric in enumerate(['mAP','frozen_F1']):
  ax=axes[i,j];ax.axhline(0,color='.55',lw=.8)
  for x,f in enumerate(FAMILIES):
   v=family_deltas[(family_deltas.comparison==comparison)&(family_deltas.family==f)].sort_values('seed')[f'delta_{metric}_pp'].to_numpy()
   for seed,y in enumerate(v):ax.scatter(x+jitter[seed],y,marker=seed_markers[seed],s=25,color=COLORS[comparison.split('-')[0]],zorder=3)
   ax.errorbar(x+.20,v.mean(),yerr=v.std(ddof=1),fmt='D',color='black',ms=3.5,capsize=2,zorder=4)
  ax.set_xticks(range(5),['Blur','Dark','Rotate','Occlude','JPEG']);ax.set_xlim(-.4,4.5)
  ax.set_title(f'({chr(97+2*i+j)}) {comparison}: {TITLES[metric]}',loc='left')
  ax.set_ylabel('Difference (percentage points)');ax.grid(axis='y',color='.92',lw=.6)
export(fig,'fig03_family_effects','Primary paired effects within each corruption family, averaging four severities within each seed before taking differences. Colored points show individual seeds; black diamonds and bars show mean and sample SD across paired seeds. Family-level analyses are descriptive and exploratory; no multiple-comparison-adjusted inference is made.')

fig,axes=plt.subplots(2,2,figsize=(8.4,7.5),layout='constrained')
attr_names=[a.replace('_',' ') for a in attribute_order]
maxabs=max(abs(attr_deltas[c]).max() for c in ['delta_AP_pp','delta_F1_pp'])
limit=math.ceil(maxabs)
# Common scale across both contrasts and metrics; keep zero centered.
for i,comparison in enumerate(['N1-N0','MTL-A']):
 for j,col in enumerate(['delta_AP_pp','delta_F1_pp']):
  ax=axes[i,j]
  pivot=attr_deltas[attr_deltas.comparison==comparison].pivot(index='attribute',columns='seed',values=col).reindex(attribute_order)[SEEDS]
  im=ax.imshow(pivot.to_numpy(),aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
  ax.set_xticks(range(3),['Seed 0','Seed 1','Seed 2']);ax.set_yticks(range(24),attr_names,fontsize=7)
  ax.set_title(f'{comparison}: {"AP" if j==0 else "Frozen F1"}',loc='left')
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.45,label='Corrupted mean difference (pp)',pad=.02)
export(fig,'fig04_attribute_effects','Per-attribute paired differences for N1-N0 and MTL-A. Each cell averages the same 20 corrupted test conditions for one attribute and seed. Red denotes a positive difference and blue a negative difference; all panels share the same symmetric scale. These post-hoc diagnostics do not establish attribute-specific causal effects or statistical significance.')

fig,axes=plt.subplots(1,2,figsize=(7.4,3.0),layout='constrained')
for model in ['N0','N1','MTL']:
 for seed in SEEDS:
  h=histories[(model,seed)];epochs=[r['epoch'] for r in h]
  style=['-','--',':'][seed]
  axes[0].plot(epochs,[100*r['clean_validation_mAP'] for r in h],ls=style,color=COLORS[model],label=model if seed==0 else None)
  selected=max(h,key=lambda r:r['clean_validation_mAP'])
  axes[0].scatter(selected['epoch'],100*selected['clean_validation_mAP'],marker=seed_markers[seed],s=20,color=COLORS[model],zorder=4)
  axes[1].plot(epochs,[r['attribute_BCE'] for r in h],ls=style,color=COLORS[model])
axes[0].set_ylabel('Clean-validation mAP (%)');axes[1].set_ylabel('Training attribute BCE')
for ax in axes:ax.set_xlabel('Epoch');ax.set_xticks(range(1,7));ax.grid(axis='y',color='.92',lw=.6)
axes[0].legend(frameon=False,ncol=3,fontsize=8)
export(fig,'fig05_training_curves','Training and checkpoint selection for the nine new runs. Colors identify models; solid/dashed/dotted lines identify seeds 0/1/2. Markers indicate the clean-validation-mAP-selected checkpoints. All N0/N1 runs stopped at epoch 4 with epoch 2 selected; MTL stopped at epochs 5/5/6 with epochs 3/3/4 selected. Right panel shows attribute BCE only, not the combined MTL loss.')

AUDIT=dict(status='PASS',source_files=len(read(ROOT/'source_manifest.json')),new_runs=9,baseline_runs=3,
    condition_rows=len(conditions),attribute_rows=len(attrs),condition_json_checks=raw_checks,
    samples_per_condition=19962,new_model_sample_evaluations=9*21*19962,
    actual_new_training_epochs=int(training[training.model!='A'].epochs.sum()),lambda_value=frozen['mtl_lambda'],
    checks=['Downloaded source byte hashes','All 12 x 21 conditions present','24 attribute values match raw per-run JSON',
    'Macro metrics equal attribute means','Archive CSV summaries and paired contrasts independently reproduced',
    'Nine completion, checkpoint-reference and threshold/test byte-hash chains match',
    'Three old A JSON artifacts match baseline binding','Best epochs match all histories; nine new runs stop at the first eligible epoch','Frozen lambda reproduced from four train batches'],
    limits=['Historical A_seed0 ran 6 epochs although the current rule would stop after epoch 4; both select epoch 2. RQ4 early stopping was introduced after training began. This is a protocol-history difference, not a reason to retrain or discard the baseline.',
    'Checkpoint binaries were not downloaded or independently rehashed; only their recorded SHA256 references were cross-checked.',
    'Predictions/logits and sample-level labels were not downloaded/exported by these RQ5 summary artifacts. Metrics were reconciled, not recomputed from predictions.',
    'Frozen-input preflight covers one image over 21 conditions; full runtime per-image checks are implementation-recorded, not rerun here.',
    'Same nominal seeds do not guarantee identical RNG trajectories. Historical baseline environment includes a Python patch-version difference. Historical A_seed0 ran six epochs rather than the four required by the current stopping rule, with epoch 2 selected in either case.',
    'Three seeds and correlated corruption conditions do not support treating conditions or attributes as independent replicates.'])
(ROOT/'audit_report.json').write_text(json.dumps(AUDIT,indent=2)+'\n')
(ROOT/'figure_captions.md').write_text('\n\n'.join(f'## {name}\n\n{text}' for name,text in CAPTIONS.items())+'\n')
# Data-driven excerpts, using paired sample SD, not SD of independent models.
P=paired_summary
insights=[]
for comparison in ['N1-N0','MTL-A','N0-A']:
 for scope in ['clean','corrupted']:
  row=P.loc[(comparison,scope)]
  vals=[]
  for metric in ['mAP','frozen_F1']:
   name=f'delta_{metric}_pp';v=paired[(paired.comparison==comparison)&(paired.scope==scope)][name]
   vals.append(f"{TITLES[metric]} {row[(name,'mean')]:+.4f} ± {row[(name,'std')]:.4f} pp ({int((v>0).sum())}/3 positive seeds)")
  insights.append(f'- {comparison}, {scope}: '+ '; '.join(vals)+'.')
(ROOT/'paper_findings_en.md').write_text('''# RQ5 evidence and manuscript wording

## Result summary

'''+ '\n'.join(insights)+'''

## Suggested Results paragraph

Matching the nonlinear classification head did not establish a consistent benefit from predicted five-point geometry. On the corrupted-test average, N1 and N0 had nearly identical mAP, while N1 had lower frozen-threshold macro-F1 on average. Paired effects changed sign across seeds. Auxiliary landmark supervision also did not provide a uniform improvement over the historical RGB-only baseline: MTL reduced corrupted mAP in all three seeds, while its mean frozen-threshold F1 increased with an inconsistent seed-level effect. The nonlinear RGB-only head itself remained close to the linear baseline on corrupted averages. These findings do not support a general robustness benefit from either intervention under the tested protocol.

## Interpretation limits

F1 and AP measure different aspects of predictions. Their divergent changes do not by themselves identify calibration, ranking, or a causal mechanism. The experiments do not establish that five landmarks contain no useful information, or rule out other fusion designs or auxiliary-loss choices. MTL used one fixed training-only scale rule (lambda=1.89811), not a sweep. RQ5 is a test-informed follow-up on the same benchmark; avoid presenting it as untouched held-out confirmation. Family and attribute plots are exploratory. Error bars are sample SD across three seeds, not confidence intervals. Results are scoped to the measured CelebA 24 attributes, ResNet18, five-point geometry, frozen conditions and budget.

## Provenance and verification

Read audit_report.json and source_manifest.json. Macro metrics and archive summaries were independently reconciled against per-run aggregate JSON. Checkpoint binary bytes and per-image model predictions were not independently evaluated in this analysis. No existing model weights, lambda, thresholds, test scores or paper files were modified.
''')
(ROOT/'README_zh.md').write_text('''# RQ5 论文证据包

## 内容与单位

- figures/：5 张图，每张提供可编辑 SVG、矢量 PDF、300 dpi PNG。
- tables/：重新计算的数据汇总、逐 seed 配对差值、分腐蚀族/属性数据、训练摘要、MTL NME，以及论文 Markdown/LaTeX 表格。
- raw/：从 Drive 下载的原始 CSV、JSON 和九个新 runs、三个旧 A runs 的记录；未下载权重。
- source_manifest.json：每个下载文件的 Drive ID、来源链接、修改时间和 SHA256。
- audit_report.json：核对项目及验证边界。
- paper_findings_en.md / figure_captions.md：英文论文结论草稿和图注。
- scripts/build_evidence.py：可重建汇总与图表的分析脚本，不运行模型。

mAP/F1 原始数据范围为 0-1；主表以百分数显示。差值统一为百分点 pp。
先在每个 seed 内等权平均 20 个非 clean 条件，再计算三 seed 的 mean 和 sample SD (ddof=1)。
配对误差条采用差值本身的样本标准差，不能用模型各自标准差代替，也不是置信区间。
clean 与 corrupted 分开；不从条件数或属性数扩大独立样本量。

## 主要判断

N1 相对 N0 没有稳定总体收益。MTL 相对 A 的腐蚀 mAP 在三个 seed 中都下降，
冻结 F1 平均提高但并非三个 seed 一致，不能称为全面改善。按条件和属性的差异只作探索性解释。
新实验均已早停：N0/N1 为 4 轮，MTL 为 5、5、6 轮，新训练合计 40 轮。旧 A_seed0 实际训练 6 轮，而当前早停规则会在第 4 轮触发；两者均选择第 2 轮权重，应在协议差异中披露。

## 复现

python scripts/build_evidence.py

依赖 numpy、pandas、matplotlib、reportlab。图表为从原始汇总记录计算得到的静态科学图。
脚本会重新核对来源 hash、输出汇总表和五张图；PDF 综合报告由 report builder 另外生成。
原始资产未修改。当前论文尚未写入本轮结果。
''')
print(json.dumps(AUDIT,indent=2))
print('\n'.join(insights))
