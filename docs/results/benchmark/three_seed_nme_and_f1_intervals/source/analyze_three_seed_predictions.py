#!/usr/bin/env python3
"""CPU-only audit, NME subgroups and paired image-bootstrap F1 intervals.

No images, checkpoints, model inference or threshold fitting are used.
Intervals are pointwise percentile intervals conditional on saved trained models.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import platform
import time
import zipfile

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

MODELS = ('r18', 'r50', 'm3', 'oracle')
RUNS = tuple(f'{m}_seed{s}' for s in range(3) for m in MODELS)
PAIRS = (('m3', 'r18'), ('m3', 'r50'), ('oracle', 'm3'), ('r50', 'r18'))
RULES = ('frozen_thresholds', 'fixed_0_5')
GROUPS = ('nme_le_5_percent', 'nme_5_to_10_percent', 'nme_10_to_20_percent',
          'nme_gt_20_percent', 'detection_failure')
ARCHIVE_SHA = '4c0cece5d688bbe87bb490c7f4c2429059a6ab40dae9c84b5cfd19ecb4baf45c'

def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        check(path.read_bytes() == data, f'Existing file differs; preserved: {path}')
        return
    temporary = path.with_name(path.name+'.partial')
    temporary.write_bytes(data)
    temporary.replace(path)


def write_csv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, stream.getvalue().encode())


def sigmoid(values):
    values = values.astype(np.float64)
    p = np.empty_like(values)
    positive = values >= 0
    p[positive] = 1/(1+np.exp(-values[positive]))
    ex = np.exp(values[~positive])
    p[~positive] = ex/(1+ex)
    return p


def read_shard(path, expected_hash, expected_binding, condition):
    data = path.read_bytes()
    check(sha(data) == expected_hash, f'Shard SHA256 mismatch: {path}')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        members = set(names)
        standard = {'manifest.json', 'records.json', 'predictions.npz'}
        oracle_schema = {'manifest.json', 'predictions.npz'}
        check(len(names) == len(members) and members in (standard, oracle_schema), f'Shard members differ: {path}')
        is_oracle = members == oracle_schema
        manifest = json.loads(archive.read('manifest.json'))
        check(manifest['binding_sha256'] == expected_binding, 'Shard execution binding mismatch')
        check(manifest['condition'] == condition, 'Shard condition mismatch')
        check(set(manifest['files']) == members - {'manifest.json'}, 'Shard payload names differ')
        parts = {name: archive.read(name) for name in manifest['files']}
        for name, payload in parts.items():
            check(sha(payload) == manifest['files'][name], f'Inner hash mismatch: {path}/{name}')
    with np.load(io.BytesIO(parts['predictions.npz']), allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if is_oracle:
        # Day21 stores pairing metadata in NPZ arrays, not records.json.
        expected_keys = {'filenames', 'targets', 'oracle_logits', 'geometry_valid',
                         'gt_normalized_points', 'gt_pixels', 'final_rgb_sha256'}
        check(set(arrays) == expected_keys, 'Day21 Oracle NPZ schema differs')
        n = len(arrays['filenames'])
        check(arrays['filenames'].shape == (n,) and arrays['filenames'].dtype.kind == 'U', 'Invalid Oracle names')
        check(arrays['targets'].shape == (n,24) and arrays['targets'].dtype == np.float32, 'Invalid Oracle targets')
        check(np.isin(arrays['targets'], [0,1]).all(), 'Oracle targets are not binary')
        check(arrays['oracle_logits'].shape == (n,24) and arrays['oracle_logits'].dtype == np.float32 and np.isfinite(arrays['oracle_logits']).all(), 'Invalid Oracle logits')
        check(arrays['geometry_valid'].shape == (n,) and arrays['geometry_valid'].dtype == np.bool_ and arrays['geometry_valid'].all(), 'Oracle GT must be valid for every sample')
        check(arrays['gt_pixels'].shape == (n,5,2) and arrays['gt_pixels'].dtype == np.float64 and np.isfinite(arrays['gt_pixels']).all(), 'Invalid Oracle pixel GT')
        check(arrays['gt_normalized_points'].shape == (n,10) and arrays['gt_normalized_points'].dtype == np.float32 and np.isfinite(arrays['gt_normalized_points']).all(), 'Invalid Oracle normalized GT')
        check(arrays['final_rgb_sha256'].shape == (n,) and arrays['final_rgb_sha256'].dtype.kind == 'U', 'Invalid Oracle RGB hashes')
        check(all(len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in arrays['final_rgb_sha256'].tolist()), 'Malformed Oracle RGB hash')
        check(isinstance(manifest.get('reference_shard_sha256'),str) and len(manifest['reference_shard_sha256'])==64, 'Missing Oracle reference-shard binding')
        # Create a read-only metadata view from checksum-verified NPZ arrays.
        records = [dict(sample_id=str(arrays['filenames'][i]), condition=condition['condition'],
                        severity=condition['severity'], final_rgb_sha256=str(arrays['final_rgb_sha256'][i]),
                        gt_pixels=arrays['gt_pixels'][i].tolist()) for i in range(n)]
    else:
        records = json.loads(parts['records.json'])
    return arrays, records, manifest


def metric_values(targets, probabilities, thresholds):
    pred = probabilities >= np.asarray(thresholds)
    tp = (pred & (targets == 1)).sum(axis=0)
    denom = pred.sum(axis=0) + targets.sum(axis=0)
    f1 = np.divide(2*tp, denom, out=np.zeros(24, dtype=float), where=denom > 0)
    positives = targets.sum(axis=0)
    ap = np.array([float(average_precision_score(targets[:,i], probabilities[:,i]))
                   if positives[i] else 0.0 for i in range(24)])
    return f1, ap, positives


def setup(project, output, repetitions, bootstrap_seed, source_code):
    project, output = Path(project), Path(output)
    check(repetitions >= 100, 'Use at least 100 bootstrap replicates')
    base = project/'runs/benchmark'
    multi = base/'multiseed_seed1_seed2'
    archive_data = (multi/'seed1_seed2_test_results_archive.zip').read_bytes()
    check(sha(archive_data) == ARCHIVE_SHA, 'Input archive differs from the completed frozen test archive')
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        manifest = json.loads(archive.read('archive_manifest.json'))
        for name, digest in manifest['files'].items():
            check(sha(archive.read(name)) == digest, f'Archive hash mismatch: {name}')
        new_report = json.loads(archive.read('multiseed_benchmark_report.json'))
        oracle_report = json.loads(archive.read('source/seed0_oracle_report.json'))
        binding = json.loads(archive.read('execution_binding.json'))
    check(new_report['full_benchmark_complete'] is True, 'Formal benchmark incomplete')
    check(new_report['threshold_fitting_performed'] is False, 'Unexpected threshold fitting')
    old_reports, hashes = {}, {}
    for day in ('day19','day20'):
        for name, digest in binding['source_reports'][day].items():
            data = (base/day/name).read_bytes()
            check(sha(data) == digest, f'Seed0 report/binding differs: {day}/{name}')
            hashes[f'{day}/{name}'] = digest
            if name.endswith('_benchmark_report.json'):
                old_reports[day] = json.loads(data)
    first_binding = json.loads((base/'day19/execution_binding.json').read_bytes())
    partition_bytes = (project/'data/list_eval_partition.txt').read_bytes()
    check(sha(partition_bytes) == first_binding['annotation_sha256']['list_eval_partition.txt'], 'Partition file changed')
    names = [row.split()[0] for row in partition_bytes.decode('utf-8-sig').splitlines() if row.split()[1]=='2']
    check(len(names) == len(set(names)) == 19962, 'Official test IDs differ')
    attributes = new_report['attribute_names']
    check(len(attributes) == 24 and attributes == oracle_report['attribute_names'], 'Attribute order mismatch')
    attr_bytes = (project/'data/list_attr_celeba.txt').read_bytes()
    check(sha(attr_bytes) == first_binding['annotation_sha256']['list_attr_celeba.txt'], 'Attribute annotations changed')
    lines = attr_bytes.decode('utf-8-sig').splitlines()
    header = lines[1].split()
    columns = [header.index(a) for a in attributes]
    needed = set(names)
    labels = {}
    for line in lines[2:]:
        fields = line.split()
        if fields[0] in needed:
            labels[fields[0]] = [int(fields[1+i]) == 1 for i in columns]
    targets = np.asarray([labels[n] for n in names], dtype=np.uint8)
    reports = [*old_reports.values(), oracle_report, new_report]
    expected = {}
    for report in reports:
        check(report['attribute_names'] == attributes, 'Report attribute order differs')
        for entry in report['conditions']:
            c = entry['condition']; label = f"{c['condition']}_s{c['severity']}"
            for run, metrics in entry['models'].items():
                check((label,run) not in expected, 'Duplicate expected metrics')
                expected[label,run] = metrics
    conditions = binding['conditions']
    check(len(conditions)==21 and len(expected)==252, 'Expected 21 conditions and 252 overall rows')
    # Freeze the existing threshold vectors and check they do not change with corruption.
    thresholds = {}
    for run in RUNS:
        thresholds[run] = expected['clean_s0',run]['frozen_thresholds']['thresholds']
        check(len(thresholds[run])==24, 'Threshold count differs')
        for condition in conditions:
            label = f"{condition['condition']}_s{condition['severity']}"
            check(expected[label,run]['frozen_thresholds']['thresholds']==thresholds[run], 'Thresholds changed across conditions')
            check(expected[label,run]['fixed_0_5']['thresholds']==[0.5]*24, 'Fixed threshold differs')
    protocol = dict(archive_sha256=ARCHIVE_SHA, source_report_hashes=hashes,
                    analysis_source_sha256=sha(source_code.encode()), oracle_shard_schema='Day21 two members; oracle_logits and paired GT/RGB in NPZ', repetitions=repetitions,
                    bootstrap_seed=bootstrap_seed, nme_groups=list(GROUPS), models=list(RUNS),
                    numpy_version=np.__version__, python_version=platform.python_version(),
                    sklearn_version=__import__('sklearn').__version__,
                    interval='95% pointwise paired image-bootstrap percentile; no multiplicity adjustment',
                    bootstrap_unit='test image; same sampled image IDs across all models and seeds within a condition',
                    mean_interval_scope='mean across these three fixed trained seeds; excludes training-seed uncertainty',
                    seed_resampling_performed=False, threshold_fitting_performed=False,
                    model_inference_performed=False, map_confidence_intervals_computed=False)
    signature = sha(encode(protocol))
    atomic(output/'analysis_binding.json', encode(protocol))
    atomic(output/'source/analyze_three_seed_predictions.py',source_code.encode())
    return dict(project=project, output=output, base=base, multi=multi, names=names, targets=targets,
                attributes=attributes, conditions=conditions, old_reports=old_reports,
                oracle_report=oracle_report, new_report=new_report, expected=expected,
                thresholds=thresholds, repetitions=repetitions, bootstrap_seed=bootstrap_seed, signature=signature)


def load_condition(ctx, condition):
    label = f"{condition['condition']}_s{condition['severity']}"
    day = 'day19' if condition['condition'] in ('clean','blur','brightness') else 'day20'
    old_report = ctx['old_reports'][day]
    parts = {run: [] for run in RUNS}
    group_parts, shard_audit = [], []
    maximum_nme_difference = 0.0
    for start in range(0,19962,256):
        stop = min(start+256,19962)
        relative = f'shards/{label}/{start:06d}_{stop:06d}.zip'
        old, records, old_manifest = read_shard(ctx['base']/day/relative, old_report['shard_sha256'][relative], old_report['execution_binding_sha256'],condition)
        oracle, oracle_records, oracle_manifest = read_shard(ctx['base']/'day21'/relative, ctx['oracle_report']['shard_sha256'][relative],ctx['oracle_report']['execution_binding_sha256'],condition)
        new, new_records, new_manifest = read_shard(ctx['multi']/relative,ctx['new_report']['shard_sha256'][relative],ctx['new_report']['execution_binding_sha256'],condition)
        check(oracle_manifest['reference_shard_sha256']==old_report['shard_sha256'][relative], 'Oracle manifest reference shard differs')
        check(new_manifest['source_shard_sha256']==old_report['shard_sha256'][relative], 'Multiseed source shard differs')
        reference_key = f'{day}/{relative}'
        check(ctx['oracle_report']['reference_shard_sha256'][reference_key]==old_report['shard_sha256'][relative], 'Oracle source shard differs')
        expected_names = ctx['names'][start:stop]
        for arrays in (old, oracle, new):
            check(arrays['filenames'].tolist()==expected_names, 'Paired image order differs')
            np.testing.assert_array_equal(arrays['targets'],ctx['targets'][start:stop])
        check(len(records)==len(oracle_records)==len(new_records)==stop-start, 'Record counts differ')
        np.testing.assert_allclose(oracle['gt_normalized_points'],new['gt_normalized_points'],rtol=0,atol=1e-6)
        np.testing.assert_array_equal(old['geometry_valid'],new['geometry_valid'])
        np.testing.assert_array_equal(old['predicted_normalized_points'],new['predicted_normalized_points'])
        groups=[]
        for i,record in enumerate(records):
            other, ore = new_records[i], oracle_records[i]
            for item in (record,other,ore):
                check(item['sample_id']==expected_names[i], 'Record ID differs')
                check(item['condition']==condition['condition'] and item['severity']==condition['severity'], 'Record condition differs')
                check(item['final_rgb_sha256']==record['final_rgb_sha256'], 'Paired RGB hash differs')
                np.testing.assert_allclose(item['gt_pixels'],record['gt_pixels'],rtol=0,atol=1e-9)
            valid=record['detection']['status']=='ok'
            check(bool(old['geometry_valid'][i])==valid, 'Detection mask differs')
            check(other['detection']==record['detection'] and other['nme']==record['nme'], 'Detector records differ')
            if valid:
                gt=np.asarray(record['gt_pixels'],dtype=float).reshape(5,2)
                points=np.asarray(record['detection']['landmarks'],dtype=float).reshape(5,2)
                eye=float(np.linalg.norm(gt[0]-gt[1]))
                check(eye>0 and np.isfinite(points).all(), 'Invalid NME inputs')
                nme=float(np.linalg.norm(points-gt,axis=1).mean()/eye)
                difference=abs(nme-record['nme'])
                check(difference<=1e-9,'NME reconstruction differs')
                maximum_nme_difference=max(maximum_nme_difference,difference)
                # Use the archived value for exact historical group boundaries.
                nme=record['nme']
                check(np.isfinite(nme) and nme>=0, 'Invalid archived NME')
                groups.append(0 if nme<=.05 else 1 if nme<=.10 else 2 if nme<=.20 else 3)
            else:
                check(record['nme'] is None, 'Failed detections must not have NME')
                check(not old['predicted_normalized_points'][i].any(), 'Invalid geometry must be zero')
                groups.append(4)
        group_parts.extend(groups)
        for run in RUNS:
            source=oracle if run=='oracle_seed0' else old if run.endswith('_seed0') else new
            key='oracle_logits' if run=='oracle_seed0' else run+'_logits'
            logits=source[key]
            check(logits.shape==(stop-start,24) and logits.dtype==np.float32 and np.isfinite(logits).all(),f'Invalid logits: {run}')
            parts[run].append(logits)
        shard_audit.append(relative)
        if (start//256+1)%20==0 or stop==19962:
            print(f'{label}: verified paired shard sets {start//256+1}/78',flush=True)
    return {run: sigmoid(np.concatenate(values)) for run,values in parts.items()}, np.asarray(group_parts), dict(paired_shard_sets=78, individual_shards=234, maximum_nme_difference=maximum_nme_difference)


def subgroup_metrics(ctx,label,probabilities,groups):
    overall, grouped, attributes = [], [], []
    targets=ctx['targets']
    for run in RUNS:
        model,seed=run.split('_seed')
        for rule in RULES:
            thresholds=ctx['thresholds'][run] if rule=='frozen_thresholds' else [0.5]*24
            f1,ap,_=metric_values(targets,probabilities[run],thresholds)
            expected=ctx['expected'][label,run][rule]
            np.testing.assert_allclose(f1,expected['per_attribute_f1'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(ap,expected['per_attribute_ap'],rtol=0,atol=1e-10)
            check(abs(f1.mean()-expected['macro_f1'])<=1e-10,'Overall F1 reproduction failed')
            check(abs(ap.mean()-expected['map'])<=1e-10,'Overall mAP reproduction failed')
            overall.append(dict(condition=label,run=run,seed=int(seed),model=model,rule=rule,macro_f1=float(f1.mean()),map=float(ap.mean())))
            for index,group in enumerate(GROUPS):
                mask=groups==index
                count=int(mask.sum())
                common=dict(condition=label,run=run,seed=int(seed),model=model,rule=rule,group=group,samples=count)
                if count:
                    gf,ga,pos=metric_values(targets[mask],probabilities[run][mask],thresholds)
                    defined=pos>0
                    grouped.append(dict(**common,macro_f1_all_24=float(gf.mean()),map_all_24_zero_if_no_positives=float(ga.mean()),
                                        map_positive_attributes_only=float(ga[defined].mean()) if defined.any() else None,
                                        attributes_without_positives=int((~defined).sum())))
                    for col,attribute in enumerate(ctx['attributes']):
                        attributes.append(dict(**common,attribute=attribute,positives=int(pos[col]),f1=float(gf[col]),
                                               ap=float(ga[col]) if defined[col] else None))
                else:
                    grouped.append(dict(**common,macro_f1_all_24=None,map_all_24_zero_if_no_positives=None,
                                        map_positive_attributes_only=None,attributes_without_positives=24))
                    for attribute in ctx['attributes']:
                        attributes.append(dict(**common,attribute=attribute,positives=0,f1=None,ap=None))
    return overall,grouped,attributes


def bootstrap_f1(ctx,label,probabilities):
    targets=ctx['targets'].astype(bool)
    # Matrix products count TP and predicted positives for all models/rules.
    columns=[]
    for rule in RULES:
        for run in RUNS:
            threshold=ctx['thresholds'][run] if rule=='frozen_thresholds' else [0.5]*24
            pred=probabilities[run]>=threshold
            columns.extend([(pred & targets).astype(np.float32),pred.astype(np.float32)])
    features=np.concatenate([targets.astype(np.float32),*columns],axis=1)
    n=len(targets); b=ctx['repetitions']
    seed=int.from_bytes(hashlib.sha256(f"{ctx['bootstrap_seed']}:{label}".encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    distributions=np.empty((b,2,len(RUNS)),dtype=float)
    started=time.perf_counter()
    with threadpool_limits(limits=2):
        for start in range(0,b,16):
            batch=min(16,b-start)
            weights=np.empty((batch,n),dtype=np.float32)
            for i in range(batch):
                weights[i]=np.bincount(rng.integers(n,size=n),minlength=n)
            counts=(weights@features).astype(np.float64)
            positive_counts=counts[:,:24]
            for rule_index in range(2):
                for run_index in range(len(RUNS)):
                    offset=24+(rule_index*len(RUNS)+run_index)*48
                    tp,pred=counts[:,offset:offset+24],counts[:,offset+24:offset+48]
                    denom=positive_counts+pred
                    f1=np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0)
                    distributions[start:start+batch,rule_index,run_index]=f1.astype(np.float64).mean(axis=1)
            if start%160==0 or start+batch==b:
                print(f'{label}: paired bootstrap {start+batch}/{b}; {(time.perf_counter()-started)/60:.1f} min',flush=True)
    result=[]
    for rule_index,rule in enumerate(RULES):
        for a,bmodel in PAIRS:
            deltas=[]
            point=[]
            for seed in range(3):
                arun,brun=f'{a}_seed{seed}',f'{bmodel}_seed{seed}'
                delta=100*(distributions[:,rule_index,RUNS.index(arun)]-distributions[:,rule_index,RUNS.index(brun)])
                observed=100*(ctx['expected'][label,arun][rule]['macro_f1']-ctx['expected'][label,brun][rule]['macro_f1'])
                deltas.append(delta);point.append(observed)
            for index in range(4):
                delta=deltas[index] if index<3 else np.mean(deltas,axis=0)
                observed=point[index] if index<3 else float(np.mean(point))
                lower,upper=np.quantile(delta,[.025,.975])
                result.append(dict(condition=label,rule=rule,comparison=f'{a}_minus_{bmodel}',
                                   scope=f'seed{index}' if index<3 else 'mean_of_three_fixed_trained_seeds',
                                   difference_pp=observed,ci95_low_pp=float(lower),ci95_high_pp=float(upper),
                                   bootstrap_replicates=ctx['repetitions'],pointwise_unadjusted=True))
    return result


def plot_results(ctx, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=('clean_s0','rotation_s4','occlusion_s4','blur_s4')
    colors=dict(zip(MODELS,('#0072B2','#E69F00','#009E73','#CC79A7')))
    for rule in RULES:
        fig,axes=plt.subplots(1,4,figsize=(17,4.5),sharey=True)
        for axis,label in zip(axes,selected):
            payload=next(p for p in results if p['condition']==label)
            for model in MODELS:
                mean,sd=[],[]
                for group in GROUPS:
                    values=[r['macro_f1_all_24'] for r in payload['group_metrics']
                            if r['model']==model and r['rule']==rule and r['group']==group]
                    mean.append(float(np.mean(values)) if values[0] is not None else np.nan)
                    sd.append(float(np.std(values,ddof=1)) if values[0] is not None else np.nan)
                axis.errorbar(range(5),mean,yerr=sd,marker='o',ms=3,label=model.upper(),color=colors[model],capsize=2)
            counts=[payload['group_counts'].get(g,0) for g in GROUPS]
            axis.set(title=label,xticks=range(5),xticklabels=[f'{g}\nn={n}' for g,n in zip(('<=5%','5-10%','10-20%','>20%','Failure'),counts)],ylim=(0,1))
            axis.tick_params(axis='x',labelsize=8)
            axis.grid(alpha=.2)
        axes[0].set_ylabel('Macro F1 (all 24 attributes)')
        axes[-1].legend(fontsize=8)
        fig.suptitle(f'{rule}: mean +/- sample SD across 3 seeds; groups may differ in prevalence')
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(ctx['output']/f'nme_groups_{rule}.{ext}',dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    labels=[p['condition'] for p in results]
    for axis,rule in zip(axes,RULES):
        rows=[next(r for r in p['intervals'] if r['rule']==rule and r['comparison']=='m3_minus_r18' and r['scope']=='mean_of_three_fixed_trained_seeds') for p in results]
        for i,row in enumerate(rows):
            axis.plot([row['ci95_low_pp'],row['ci95_high_pp']],[i,i],color='#0072B2')
            axis.scatter([row['difference_pp']],[i],color='#0072B2',s=15)
        axis.axvline(0,color='black',lw=.8)
        axis.set(title=rule,xlabel='M3 - R18 (percentage points)',yticks=range(21),yticklabels=labels)
        axis.grid(alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Paired image-bootstrap 95% intervals: mean over 3 FIXED trained seeds\nPointwise/unadjusted; training-seed uncertainty excluded',fontsize=11)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(ctx['output']/f'm3_r18_fixed_seed_image_intervals.{ext}',dpi=160)
    plt.close(fig)


def run_analysis(ctx):
    all_results=[]
    for condition in ctx['conditions']:
        label=f"{condition['condition']}_s{condition['severity']}"
        path=ctx['output']/'conditions'/f'{label}.json'
        if path.exists():
            result=json.loads(path.read_bytes())
            check(result['analysis_binding_sha256']==ctx['signature'],'Saved analysis binding differs')
            payload=result['payload']
            check(sha(encode(payload))==result['payload_sha256'],'Saved analysis checksum differs')
            print(label+': verified saved analysis; skipped shard reads and bootstrap',flush=True)
        else:
            probabilities,groups,audit=load_condition(ctx,condition)
            overall,grouped,attributes=subgroup_metrics(ctx,label,probabilities,groups)
            print(label+': both F1 rules and mAP reproduced for all 12 runs',flush=True)
            intervals=bootstrap_f1(ctx,label,probabilities)
            payload=dict(condition=label,audit=audit,group_counts=dict(Counter(GROUPS[int(i)] for i in groups)),
                         overall=overall,group_metrics=grouped,group_attribute_metrics=attributes,intervals=intervals)
            result=dict(analysis_binding_sha256=ctx['signature'],payload_sha256=sha(encode(payload)),payload=payload)
            atomic(path,encode(result))
            del probabilities,groups
            print(label+': analysis saved',flush=True)
        all_results.append(payload)
    for key in ('overall','group_metrics','group_attribute_metrics','intervals'):
        rows=[row for payload in all_results for row in payload[key]]
        write_csv(ctx['output']/f'{key}.csv',rows)
    # Average subgroup scores across seeds without treating seeds as extra images.
    grouped={}
    for payload in all_results:
        for row in payload['group_metrics']:
            key=(row['condition'],row['model'],row['rule'],row['group'])
            grouped.setdefault(key,[]).append(row)
    summary=[]
    for key,rows in grouped.items():
        check(len(rows)==3 and len({r['samples'] for r in rows})==1,'Subgroup membership differs across seeds')
        vals=[r['macro_f1_all_24'] for r in rows]
        summary.append(dict(condition=key[0],model=key[1],rule=key[2],group=key[3],samples_per_seed=rows[0]['samples'],
                            mean_f1=float(np.mean(vals)) if vals[0] is not None else None,
                            sd_f1=float(np.std(vals,ddof=1)) if vals[0] is not None else None))
    write_csv(ctx['output']/'group_three_seed_summary.csv',summary)
    plot_results(ctx, all_results)
    notes='''# Three-seed NME subgroup and paired F1 analysis

NME groups: <=5%, (5%,10%], (10%,20%], >20%, detection failure.
All models are compared on the same detector-defined groups, including Oracle.
F1 uses all 24 attributes; undefined F1 is zero for nonempty groups. Empty groups remain missing.
AP is missing for attributes without positives. Both the positive-attribute-only mAP and
an all-24 convention assigning zero to absent-positive AP are explicitly labeled.
Do not compare tiny or different-prevalence subgroups as causal effects of NME.

Intervals: 95% percentile paired test-image bootstrap. Whole images, including all 24
labels, are resampled with replacement. The same resample is used across all models
and seeds within each condition. Intervals are provided for each fixed seed and the
mean difference across the three fixed trained seeds. No training-seed resampling is
performed. These intervals exclude training-run uncertainty and must not be interpreted
as confidence intervals over arbitrary future training seeds. Seed0 environment
changes are a further limitation. Intervals are exploratory, pointwise and unadjusted
for multiple comparisons; no p-values or blanket significance claims are produced.
mAP is reproduced and tabulated; mAP confidence intervals are not computed.

Existing inference thresholds are reused unchanged. No model inference, retraining,
threshold fitting or detector execution occurs. Resume verifies saved analysis payloads
and their frozen binding, then skips completed conditions; skipped conditions do not
re-read source shards. Remove no files and do not run concurrent writers.
'''
    atomic(ctx['output']/'README.md',notes.encode())
    audit=dict(complete=True,conditions=21,paired_shard_sets_verified=sum(x['audit']['paired_shard_sets'] for x in all_results),
               individual_shards_verified=sum(x['audit']['individual_shards'] for x in all_results),
               overall_rule_rows=504,group_metric_rows=2520,group_attribute_rows=60480,interval_rows=672,
               analysis_binding_sha256=ctx['signature'],training=False,inference=False,threshold_fitting=False)
    check(len([r for p in all_results for r in p['intervals']])==672,'Interval count differs')
    atomic(ctx['output']/'completion_audit.json',encode(audit))
    # Exclude bulky per-condition JSON from the transfer archive; CSV retains every result.
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(ctx['output'].rglob('*')):
            if file.is_file() and file.suffix in ('.csv','.md','.py','.json','.png','.pdf') and 'conditions' not in file.relative_to(ctx['output']).parts:
                info=zipfile.ZipInfo(file.relative_to(ctx['output']).as_posix(),date_time=(2026,1,1,0,0,0))
                z.writestr(info,file.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
    archive=ctx['output']/'three_seed_nme_and_f1_intervals_archive.zip'
    atomic(archive,bundle.getvalue())
    print('Three-seed NME subgroups and paired F1 intervals: COMPLETE')
    print('Archive:',archive)
    print('Archive SHA256:',sha(bundle.getvalue()))
    return archive
