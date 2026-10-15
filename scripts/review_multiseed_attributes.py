#!/usr/bin/env python3
"""Read-only project inventory and three-seed attribute comparisons.

Run from the repository: python scripts/review_multiseed_attributes.py --project .
Dependencies: numpy, matplotlib. No training, inference, fitting or file deletion.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=('r18','r50','m3','oracle')
SEEDS=(0,1,2)
FAMILIES=('blur','brightness','rotation','occlusion','jpeg')
CONDITIONS=('clean_s0',)+tuple(f'{f}_s{s}' for f in FAMILIES for s in range(1,5))
METRICS=('f1_frozen','f1_at_0.5','ap')
COMPARISONS=(('oracle','m3'),('m3','r18'))


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.read_bytes()==data,f'Existing output differs; preserved: {path}. Choose a new --output directory.')
        return
    temporary=path.with_name(path.name+'.partial')
    temporary.write_bytes(data)
    temporary.replace(path)


def save_csv(path,rows):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_once(path,stream.getvalue().encode())


def read_inputs(project):
    base=project/'docs/results/benchmark'
    multi=base/'multiseed_seed1_seed2'
    binding_bytes=(multi/'execution_binding.json').read_bytes()
    binding=json.loads(binding_bytes)
    source_hashes={}
    new_bytes=(multi/'multiseed_benchmark_report.json').read_bytes()
    new=json.loads(new_bytes)
    require(sha(binding_bytes)==new['execution_binding_sha256'],'Multiseed execution binding hash mismatch')
    archive_manifest=json.loads((multi/'archive_manifest.json').read_bytes())
    for name in ('multiseed_benchmark_report.json','source/seed0_oracle_report.json'):
        data=(multi/name).read_bytes()
        require(sha(data)==archive_manifest['files'][name],f'Archived file hash differs: {name}')
        source_hashes[str((multi/name).relative_to(project))]=sha(data)
    oracle=json.loads((multi/'source/seed0_oracle_report.json').read_bytes())
    reports=[]
    for day in ('day19','day20'):
        for name,expected in binding['source_reports'][day].items():
            path=base/day/name;data=path.read_bytes()
            require(sha(data)==expected,f'Historical input hash differs: {path}')
            source_hashes[str(path.relative_to(project))]=expected
            if name.endswith('_benchmark_report.json'):reports.append(json.loads(data))
    require(new['full_benchmark_complete'] is True,'Formal test benchmark incomplete')
    reports.extend([oracle,new])
    attributes=new['attribute_names'];require(len(attributes)==len(set(attributes))==24,'Attribute list differs')
    records={};thresholds={};positive_counts=None
    for report in reports:
        require(report['attribute_names']==attributes,'Attribute order differs across reports')
        require(report['threshold_fitting_performed'] is False,'Unexpected test threshold fitting')
        for condition in report['conditions']:
            c=condition['condition'];label=f"{c['condition']}_s{c['severity']}"
            for run,metrics in condition['models'].items():
                model,seed=run.split('_seed');seed=int(seed)
                frozen,fixed=metrics['frozen_thresholds'],metrics['fixed_0_5']
                require(frozen['sample_count']==fixed['sample_count']==19962,'Sample count mismatch')
                require(frozen['per_attribute_ap']==fixed['per_attribute_ap'],'AP changed with F1 threshold rule')
                require(fixed['thresholds']==[.5]*24,'Fixed threshold mismatch')
                if run in thresholds:require(thresholds[run]==frozen['thresholds'],'Frozen thresholds vary across conditions')
                thresholds[run]=frozen['thresholds']
                require(len(thresholds[run])==24,'Threshold count mismatch')
                counts=frozen['positive_counts']
                require(counts==fixed['positive_counts'],'Target counts differ between rules')
                if positive_counts is not None:require(counts==positive_counts,'Test target positive counts differ')
                positive_counts=counts
                for metric,values in (('f1_frozen',frozen['per_attribute_f1']),('f1_at_0.5',fixed['per_attribute_f1']),('ap',frozen['per_attribute_ap'])):
                    require(len(values)==24 and all(np.isfinite(v) and 0<=v<=1 for v in values),'Invalid attribute metric')
                    expected=frozen['map'] if metric=='ap' else frozen['macro_f1'] if metric=='f1_frozen' else fixed['macro_f1']
                    require(abs(float(np.mean(values))-expected)<=1e-12,'Attribute-to-overall reproduction failed')
                for i,attribute in enumerate(attributes):
                    key=(label,model,seed,attribute)
                    require(key not in records,'Duplicate attribute row')
                    records[key]=dict(condition=label,model=model,seed=seed,attribute=attribute,samples=19962,
                                      positives=counts[i],f1_frozen=frozen['per_attribute_f1'][i],
                                      **{'f1_at_0.5':fixed['per_attribute_f1'][i],'ap':frozen['per_attribute_ap'][i]})
    expected={(c,m,s,a) for c in CONDITIONS for m in MODELS for s in SEEDS for a in attributes}
    require(set(records)==expected,'Expected exactly 6048 complete attribute rows')
    return attributes,records,source_hashes


def inventory(project):
    folders=[];errors=[];notebook_hashes=defaultdict(list)
    # Metadata-only scan. Skip internal environments and follow no symbolic links.
    for top in sorted(project.iterdir()):
        if top.name.startswith('.'):
            continue
        if top.is_symlink():
            folders.append(dict(path=top.name,kind='symlink',files=None,bytes=None));continue
        if top.is_file():
            folders.append(dict(path=top.name,kind='file',files=1,bytes=top.stat().st_size));continue
        count=0;total=0
        for directory,subdirs,files in os.walk(top,followlinks=False,onerror=lambda e:errors.append(str(e))):
            subdirs[:]=[d for d in subdirs if not d.startswith('.') and not (Path(directory)/d).is_symlink()]
            for filename in files:
                p=Path(directory)/filename
                if filename.startswith('.') or p.is_symlink():continue
                try:total+=p.stat().st_size;count+=1
                except OSError as e:errors.append(str(e))
        folders.append(dict(path=top.name,kind='directory',files=count,bytes=total))
    notebooks=project/'notebooks'
    if notebooks.is_dir():
        for p in sorted(notebooks.glob('*.ipynb')):
            try:notebook_hashes[sha(p.read_bytes())].append(str(p.relative_to(project)))
            except OSError as e:errors.append(str(e))
    git=subprocess.run(['git','status','--short'],cwd=project,capture_output=True,text=True)
    metadata=[]
    for seed in SEEDS:
        for model in MODELS:
            name=f'{model}_seed{seed}';p=project/'runs'/name
            metadata.append(dict(run=name,local_run_directory=p.is_dir(),
                                 config_json=(p/'config.json').is_file(),best_json=(p/'best.json').is_file(),
                                 history_json=(p/'history.json').is_file(),
                                 note='Local availability only; training files may intentionally remain on Drive.'))
    return dict(top_level_inventory=folders,exact_duplicate_notebooks=[v for v in notebook_hashes.values() if len(v)>1],
                local_run_record_availability=metadata,git_status=git.stdout.splitlines(),git_status_error=git.stderr.strip() if git.returncode else None,
                scan_errors=errors,files_moved=False,files_deleted=False,
                note='Only folder metadata and notebook hashes were inspected; raw images/checkpoints/prediction contents were not read.')


def compare(project,output):
    attributes,data,hashes=read_inputs(project)
    raw=[data[c,m,s,a] for c in CONDITIONS for m in MODELS for s in SEEDS for a in attributes]
    summary=[];differences=[];family=[]
    for c in CONDITIONS:
        for a in attributes:
            for model in MODELS:
                for metric in METRICS:
                    vals=[data[c,model,s,a][metric] for s in SEEDS]
                    summary.append(dict(condition=c,attribute=a,model=model,metric=metric,
                                        mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1))))
            for lhs,rhs in COMPARISONS:
                for metric in METRICS:
                    delta=[100*(data[c,lhs,s,a][metric]-data[c,rhs,s,a][metric]) for s in SEEDS]
                    differences.append(dict(condition=c,attribute=a,comparison=f'{lhs}_minus_{rhs}',metric=metric,
                                            mean_difference_pp=float(np.mean(delta)),sd_difference_pp=float(np.std(delta,ddof=1)),
                                            positive_seeds=sum(v>0 for v in delta),negative_seeds=sum(v<0 for v in delta),
                                            **{f'seed{s}_difference_pp':delta[s] for s in SEEDS}))
    for group in ('clean',*FAMILIES,'all_corruptions'):
        conditions=['clean_s0'] if group=='clean' else list(CONDITIONS[1:]) if group=='all_corruptions' else [f'{group}_s{s}' for s in range(1,5)]
        for a in attributes:
            for lhs,rhs in COMPARISONS:
                for metric in METRICS:
                    delta=[100*np.mean([data[c,lhs,s,a][metric]-data[c,rhs,s,a][metric] for c in conditions]) for s in SEEDS]
                    family.append(dict(family=group,attribute=a,comparison=f'{lhs}_minus_{rhs}',metric=metric,
                                       mean_difference_pp=float(np.mean(delta)),sd_difference_pp=float(np.std(delta,ddof=1)),
                                       positive_seeds=sum(v>0 for v in delta),negative_seeds=sum(v<0 for v in delta),
                                       **{f'seed{s}_difference_pp':float(delta[s]) for s in SEEDS}))
    output.mkdir(parents=True,exist_ok=True)
    save_csv(output/'all_seed_attribute_metrics.csv',raw)
    save_csv(output/'attribute_mean_sd.csv',summary)
    save_csv(output/'attribute_paired_differences_pp.csv',differences)
    save_csv(output/'attribute_family_differences_pp.csv',family)
    # Descriptive direction consistency; no significance claim or optimized cutoff.
    consistent=[r for r in family if r['positive_seeds']==3 or r['negative_seeds']==3]
    save_csv(output/'attributes_with_same_direction_in_all_seeds.csv',consistent)
    selected=('clean','blur','brightness','rotation','occlusion','jpeg')
    for lhs,rhs in COMPARISONS:
        fig,axes=plt.subplots(1,3,figsize=(16,9),sharey=True)
        matrices=[]
        for metric in METRICS:
            matrices.append(np.array([[next(r['mean_difference_pp'] for r in family if r['family']==g and r['attribute']==a and r['comparison']==f'{lhs}_minus_{rhs}' and r['metric']==metric) for g in selected] for a in attributes]))
        limit=max(1.,max(float(np.abs(v).max()) for v in matrices))
        for axis,metric,values in zip(axes,METRICS,matrices):
            im=axis.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
            axis.set(title=metric,xticks=range(6),xticklabels=selected,yticks=range(24),yticklabels=attributes)
            axis.tick_params(axis='x',rotation=45)
            fig.colorbar(im,ax=axis,label='Difference (percentage points)',fraction=.045)
        fig.suptitle(f'{lhs.upper()} minus {rhs.upper()}: mean attribute difference across 3 seeds\nEach corruption family averages its 4 severities; descriptive, no attribute-level inference')
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ('png','pdf'):
            # Plot renderers add timestamps; figures may be refreshed on reruns.
            fig.savefig(output/f'attribute_heatmap_{lhs}_minus_{rhs}.{ext}',dpi=180)
        plt.close(fig)
    notes=['# Three-seed per-attribute review','',
           'Complete coverage: 6048 rows = 21 conditions x 4 models x 3 seeds x 24 attributes.',
           'F1(frozen), F1@0.5 and AP are separate. mAP is the mean of the 24 attribute AP values.',
           'Comparisons: Oracle - M3 and M3 - R18, matched by seed, attribute and condition.',
           'Family differences first average the four severities within each seed, then summarize seeds.',
           'SD uses ddof=1. All-3-seed direction consistency is descriptive, not statistical significance.',
           'No test threshold fitting, retraining or inference was performed.',
           'Oracle is an independently trained GT-reference model, not a guaranteed upper bound.',
           'Oracle - M3 is not a causal detector-error effect: these are independently trained models.',
           'Seed0 environment differences and only three training seeds limit generalization.',
           'Do not select a new model/threshold using these test results and report it as an untouched test.',
           '', '## Oracle - M3: all-corruption attribute differences',
           'All attributes are reported in the predefined order, without post-hoc selection.','']
    for a in attributes:
        vals=[next(r for r in family if r['family']=='all_corruptions' and r['attribute']==a and r['comparison']=='oracle_minus_m3' and r['metric']==metric) for metric in METRICS]
        notes.append('- '+a+': '+ '; '.join(f"{r['metric']} {r['mean_difference_pp']:+.3f} pp (positive seeds {r['positive_seeds']}/3)" for r in vals))
    write_once(output/'README.md','\n'.join(notes).encode()+b'\n')
    audit=dict(complete=True,overall_attribute_rows=6048,attribute_summary_rows=len(summary),
               paired_attribute_rows=len(differences),family_comparison_rows=len(family),
               source_hashes=hashes,macro_metric_reproduction=True,positive_counts_aligned=True,
               thresholds_fixed_across_conditions=True,training=False,inference=False,threshold_fitting=False,
               attribute_confidence_intervals_computed=False)
    write_once(output/'attribute_analysis_audit.json',json_bytes(audit))
    # Refresh local inventory separately; it changes as files are added or committed.
    (output/'local_inventory.json').write_bytes(json_bytes(inventory(project)))
    print('Input hashes, target counts, attribute order and macro metric reproduction: PASS')
    print('Three-seed per-attribute analysis: COMPLETE')
    print('Attribute metric rows:',len(raw))
    print('Local inventory saved; no files moved or deleted')
    print('Saved:',output.resolve())
    print('\nOracle - M3, all corruptions, F1(frozen), all 3 seeds positive:')
    for row in family:
        if row['family']=='all_corruptions' and row['comparison']=='oracle_minus_m3' and row['metric']=='f1_frozen' and row['positive_seeds']==3:
            print(row['attribute'],f"{row['mean_difference_pp']:+.3f} pp")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args();project=args.project.resolve()
    output=args.output or project/'docs/results/benchmark/three_seed_attribute_comparison'
    compare(project,output)
