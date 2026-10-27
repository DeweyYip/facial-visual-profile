"""Evaluate frozen RQ5 checkpoints; reruns skip only bound completed conditions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import pandas as pd
from datasets.celeba_rq5 import CelebARQ5, ATTRIBUTES
from training.rq5_shared import FAMILIES, compute_metrics
from training.rq5 import load_config, load_best, predict, write_json, sha256, source_hashes


def conditions():
    yield 'clean', None, 0
    for family in FAMILIES:
        for severity in range(1,5):
            yield f'{family}_s{severity}', family, severity


def evaluate(project,images,name,seed,config,roots,device):
    model,folder = load_best(project,name,seed,config,device)
    threshold_path = folder/'clean_val_thresholds.json'
    threshold = json.loads(threshold_path.read_text())
    if threshold['attribute_order'] != ATTRIBUTES or threshold['checkpoint_sha256'] != sha256(folder/'best_model.pt'):
        raise ValueError('Frozen threshold provenance mismatch')
    binding = dict(checkpoint=sha256(folder/'best_model.pt'),thresholds=sha256(threshold_path),
                   sources=source_hashes(),evaluation_script=sha256(__file__),
                   images=str(Path(images).resolve()),benchmark_roots=[str(Path(r).resolve()) for r in roots])
    binding_path = folder/'test_binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Evaluation binding differs')
    write_json(binding_path,binding)
    path = folder/'test_metrics.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    for label,family,severity in conditions():
        if label in results:
            continue
        ds = CelebARQ5(project,images,'test',name,family,severity,roots)
        truth,prob,nme = predict(model,name,ds,config,device)
        metrics = compute_metrics(truth,prob,np.asarray(threshold['thresholds']))
        metrics['F1_at_05'] = compute_metrics(truth,prob)['macro_f1']
        if nme:
            metrics['landmark_NME'] = dict(count=len(nme),mean=float(np.mean(nme)),median=float(np.median(nme)))
        results[label] = metrics
        write_json(path,results)
        print(name,seed,label,metrics['mAP'],metrics['macro_f1'],flush=True)
    if set(results) != {label for label,_,_ in conditions()}:
        raise ValueError('Condition set differs')
    write_json(folder/'test_complete.json',dict(conditions=21))


def report(project,config):
    study = Path(project)/'runs'/config['study']
    rows, attributes = [], []
    for name in ['N0','N1','MTL','A']:
        base = Path(project)/'runs'/('rq4_training_v1' if name == 'A' else config['study'])
        for seed in config['seeds']:
            folder = base/f'{name}_seed{seed}'
            if not (folder/'test_complete.json').exists():
                raise ValueError(f'Incomplete result: {folder}')
            data = json.loads((folder/'test_metrics.json').read_text())
            if set(data) != {label for label,_,_ in conditions()}:
                raise ValueError('Result condition mismatch')
            for label,metrics in data.items():
                rows.append(dict(model=name,seed=seed,condition=label,
                    family='clean' if label=='clean' else label.rsplit('_s',1)[0],
                    mAP=metrics['mAP'],frozen_F1=metrics['macro_f1'],F1_at_05=metrics['F1_at_05']))
                for i,attribute in enumerate(ATTRIBUTES):
                    attributes.append(dict(model=name,seed=seed,condition=label,attribute=attribute,
                                           AP=metrics['AP_by_attr'][i],F1=metrics['F1_by_attr'][i]))
    table = pd.DataFrame(rows)
    table.to_csv(study/'condition_results.csv',index=False)
    pd.DataFrame(attributes).to_csv(study/'attribute_results.csv',index=False)
    measures = ['mAP','frozen_F1','F1_at_05']
    corrupt = table[table.condition!='clean'].groupby(['model','seed'])[measures].mean()
    corrupt.to_csv(study/'corruption_averages_by_seed.csv')
    corrupt.groupby('model')[measures].agg(['mean','std']).to_csv(study/'corruption_summary.csv')
    table.groupby(['model','seed','family'])[measures].mean().to_csv(study/'family_results.csv')
    contrasts = []
    for scope in ['clean','corrupted']:
        values = (table[table.condition=='clean'].set_index(['model','seed'])[measures]
                  if scope=='clean' else corrupt)
        for a,b in [('N1','N0'),('MTL','A'),('N0','A')]:
            for seed in config['seeds']:
                row = dict(comparison=f'{a}-{b}',scope=scope,seed=seed)
                row.update({f'delta_{m}_pp':100*(values.loc[(a,seed),m]-values.loc[(b,seed),m]) for m in measures})
                contrasts.append(row)
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(study/'paired_seed_contrasts.csv',index=False)
    contrasts.groupby(['comparison','scope'])[[f'delta_{m}_pp' for m in measures]].agg(['mean','std']).to_csv(study/'paired_summary.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True)
    parser.add_argument('--images')
    parser.add_argument('--config',default='configs/rq5_training_v1.json')
    parser.add_argument('--model',choices=['N0','N1','MTL'])
    parser.add_argument('--seed',type=int,choices=[0,1,2])
    parser.add_argument('--benchmark-roots',nargs='*',default=[])
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.report_only:
        report(args.project,config)
    else:
        if not args.images or args.model is None or args.seed is None:
            parser.error('Evaluation requires --images, --model and --seed')
        if args.model=='N1' and not args.benchmark_roots:
            parser.error('N1 requires original benchmark shard roots')
        evaluate(args.project,args.images,args.model,args.seed,config,args.benchmark_roots,
                 'cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    main()
