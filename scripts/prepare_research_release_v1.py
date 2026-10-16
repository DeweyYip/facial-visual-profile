#!/usr/bin/env python3
"""Assemble a local research-release index from completed archived experiments.

No training, inference, threshold fitting, Git mutation, or file deletion.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess

MODELS = ('r18', 'r50', 'm3', 'oracle')
METRICS = ('f1_frozen', 'f1_at_0.5', 'map')
ROOT = Path('docs/results/benchmark')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('.'))
    args = parser.parse_args()
    project = args.project.resolve()
    require((project/'.git').exists(), 'Run inside the project checkout')
    output = project/'docs/releases/research_v1'
    tracked_inputs = {}

    def read(relative):
        relative = Path(relative)
        data = (project/relative).read_bytes()
        tracked_inputs[relative.as_posix()] = sha(data)
        return data

    def json_read(relative):
        return json.loads(read(relative))

    # Verify the archived payloads rather than trusting completion text alone.
    for directory in ('multiseed_seed1_seed2', 'prediction_case_review', 'small_scale_reproduction'):
        prefix = ROOT/directory
        manifest = json_read(prefix/'archive_manifest.json')
        for name, digest in manifest['files'].items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe manifest path')
            require(sha(read(prefix/rel)) == digest, f'Archived member changed: {prefix/rel}')
    case_root = ROOT/'prediction_case_review'
    case_audit = json_read(case_root/'completion_audit.json')
    cases = json_read(case_root/'selected_cases.json')
    require(case_audit['complete'] and case_audit['all_rendered_inputs_verified'], 'Case review incomplete')
    require(case_audit['cases']==len(cases)==24 and case_audit['prediction_rows']==288, 'Case counts differ')
    require(len({c['filename'] for c in cases})==24, 'Unique case image count differs')
    require(Counter(c['condition'] for c in cases)=={c:6 for c in case_audit['conditions']}, 'Case condition counts differ')
    case_binding = json_read(case_root/'analysis_binding.json')
    require(sha(read(case_root/'analysis_binding.json'))==case_audit['binding_sha256'], 'Case binding differs')
    case_metadata_issue = case_binding['conditions'] != case_audit['conditions']
    require(case_metadata_issue, 'Expected original metadata issue changed; review the release note')
    require(case_binding['conditions']==[f'{n:06d}.jpg' for n in range(182638,202600)], 'Unexpected case metadata discrepancy')
    repro = json_read(ROOT/'small_scale_reproduction/reproduction_report.json')
    require(repro['execution_complete'] and repro['reproduction_passed'], 'Small replay did not pass')
    require(repro['model_sample_evaluations']==128 and repro['attribute_comparison_rows']==3072, 'Replay counts differ')
    require(len(repro['results'])==16 and all(x['replay_passed'] for x in repro['results']), 'Replay condition failure')
    require(all(all(v['changed_decisions']==0 for v in x['decisions'].values()) for x in repro['results']), 'Replay decisions changed')
    nme = json_read(ROOT/'three_seed_nme_and_f1_intervals/completion_audit.json')
    require(nme['complete'] and nme['conditions']==21 and nme['interval_rows']==672, 'NME/interval analysis incomplete')
    nme_binding = json_read(ROOT/'three_seed_nme_and_f1_intervals/analysis_binding.json')
    require(sha(read(ROOT/'three_seed_nme_and_f1_intervals/analysis_binding.json'))==nme['analysis_binding_sha256'], 'NME binding differs')
    require(nme_binding['repetitions']==1000 and not nme_binding['map_confidence_intervals_computed'], 'Statistical scope differs')
    # Reproduce the 252 CSV rows from original report metrics, including Oracle Seed0.
    reports = [json_read(ROOT/p) for p in (
        'day19/day19_benchmark_report.json', 'day20/day20_benchmark_report.json',
        'multiseed_seed1_seed2/source/seed0_oracle_report.json',
        'multiseed_seed1_seed2/multiseed_benchmark_report.json')]
    expected = {}
    for report in reports:
        for entry in report['conditions']:
            c = entry['condition']; label=f"{c['condition']}_s{c['severity']}"
            for run, value in entry['models'].items():
                require((label,run) not in expected, 'Duplicate report row')
                expected[label,run] = dict(f1_frozen=value['frozen_thresholds']['macro_f1'],
                    f1_at_0_5=value['fixed_0_5']['macro_f1'],map=value['frozen_thresholds']['map'])
    raw_csv = read(ROOT/'three_seed_comparison/all_seed_metrics.csv').decode('utf-8-sig')
    rows = list(csv.DictReader(io.StringIO(raw_csv)))
    require(len(rows)==len(expected)==252, 'Expected 252 overall metric rows')
    keys=set()
    for row in rows:
        key=(row['condition'],row['run_name']);require(key not in keys and key in expected, 'Duplicate or unexpected metric row');keys.add(key)
        require(row['model'] in MODELS and row['seed'] in ('0','1','2'), 'Invalid model or seed')
        require(row['run_name']==f"{row['model']}_seed{row['seed']}" and int(row['samples'])==19962,'Run identity or sample count differs')
        for metric in METRICS:
            value=float(row[metric]);require(math.isfinite(value) and 0<=value<=1,'Invalid metric')
            key_name='f1_at_0_5' if metric=='f1_at_0.5' else metric
            require(abs(value-expected[key][key_name])<=1e-12, f'CSV/report mismatch: {key}/{metric}')
    conditions=sorted({r['condition'] for r in rows})
    require(len(conditions)==21 and 'clean_s0' in conditions,'Condition coverage differs')
    summary=[]
    for scope in ('clean','all_20_corruptions'):
        for model in MODELS:
            for metric in METRICS:
                values=[]
                for seed in ('0','1','2'):
                    subset=[float(r[metric]) for r in rows if r['model']==model and r['seed']==seed and ((r['condition']=='clean_s0')==(scope=='clean'))]
                    require(len(subset)==(1 if scope=='clean' else 20),'Summary denominator differs')
                    values.append(statistics.mean(subset))
                summary.append(dict(scope=scope,model=model,metric=metric,mean=statistics.mean(values),
                    sample_sd=statistics.stdev(values),seed0=values[0],seed1=values[1],seed2=values[2]))
    assets = [
        ('Overall metrics', 'three_seed_comparison/all_seed_metrics.csv'),
        ('Frozen F1 curves', 'three_seed_comparison/curves_f1_frozen.png'),
        ('F1 at 0.5 curves', 'three_seed_comparison/curves_f1_at_0_5.png'),
        ('mAP curves', 'three_seed_comparison/curves_map.png'),
        ('Per-seed M3 minus R18', 'three_seed_comparison/paired_m3_r18.png'),
        ('Fixed-model image-bootstrap intervals', 'three_seed_nme_and_f1_intervals/m3_r18_fixed_seed_image_intervals.png'),
        ('NME groups, frozen', 'three_seed_nme_and_f1_intervals/nme_groups_frozen_thresholds.png'),
        ('NME groups, 0.5', 'three_seed_nme_and_f1_intervals/nme_groups_fixed_0_5.png'),
        ('M3 minus R18 attribute heatmap', 'three_seed_attribute_comparison/attribute_heatmap_m3_minus_r18.png'),
        ('Oracle minus M3 attribute heatmap', 'three_seed_attribute_comparison/attribute_heatmap_oracle_minus_m3.png'),
        ('Attribute family differences', 'three_seed_attribute_comparison/attribute_family_differences_pp.csv'),
        ('Selected cases and source records', 'prediction_case_review/selected_cases.json'),
        ('Case predictions', 'prediction_case_review/case_predictions.csv'),
        ('Replay report', 'small_scale_reproduction/reproduction_report.json'),
    ]
    for title, relative in assets:read(ROOT/relative)
    notebooks = ('multiseed_training_and_validation.ipynb','multiseed_training_and_validation_seed2.ipynb',
                 'multiseed_frozen_test_benchmark.ipynb','three_seed_nme_and_f1_intervals.ipynb',
                 'three_seed_prediction_case_review.ipynb','small_scale_inference_reproduction.ipynb')
    for name in notebooks:read(Path('notebooks')/name)
    max_error=max(r['max_absolute_logit_error'] for r in repro['results'])
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    dirty=set()
    for command in (['git','diff','--name-only'],['git','diff','--cached','--name-only']):
        dirty.update(subprocess.check_output(command,cwd=project,text=True).splitlines())
    lines=['# Research release v1','','Status: local research-result snapshot; not a published paper or a GitHub Release.','',
        '## Research question','',
        'Does concatenating an explicit five-point geometry branch improve CelebA facial-attribute robustness relative to RGB baselines?','',
        '## Frozen experimental setup','',
        '- CelebA official splits: 162770 train, 19867 validation, 19962 test; 24 attributes.',
        '- Four models: ImageNet-initialized R18, R50, R18 plus predicted MTCNN geometry (M3), and an independently trained GT-geometry reference (Oracle).',
        '- Geometry branch: 10 coordinates -> 32 -> 32; concatenate with 512 RGB features. Invalid M3 geometry is masked after the MLP; samples are retained.',
        '- Three training seeds (0, 1, 2); ten epochs per run; Adam, learning rate 0.0001, batch 64, FP32, no augmentation.',
        '- Select highest clean-validation mAP, earliest epoch on ties. Fit per-attribute F1 thresholds on clean validation only; grid 0.05..0.95, step 0.01, tie closest to 0.5 then lower.',
        '- Shared 224x224 letterbox preprocessing. Five corruption families x four severities, plus clean: 21 conditions. Same test inputs are paired across models and training seeds.',
        '- Blur sigma 1/2/3/4; brightness factors 0.8/0.6/0.4/0.2; rotation magnitude 5/10/15/20 degrees with frozen deterministic signs; lower-face square occlusion side fractions 0.1/0.2/0.3/0.4; JPEG quality 75/50/25/10.',
        '- F1(frozen), F1@0.5 and mAP remain separate. No test threshold fitting or test-based checkpoint selection.','',
        '## Descriptive summary','',
        'For corruptions, average the twenty conditions equally within each seed, then report mean +/- sample SD across the three resulting seed values. Clean is excluded from that average. SD is not a confidence interval.','',
        '| Scope | Model | F1(frozen) mean +/- SD | F1@0.5 mean +/- SD | mAP mean +/- SD |','|---|---|---|---|---|']
    for scope in ('clean','all_20_corruptions'):
        for model in MODELS:
            selected={r['metric']:r for r in summary if r['scope']==scope and r['model']==model}
            vals=[f"{selected[m]['mean']:.6f} +/- {selected[m]['sample_sd']:.6f}" for m in METRICS]
            lines.append('| '+' | '.join([scope,model,*vals])+' |')
    lines += ['', '## Supported interpretation and limits','',
        '- These three runs do not show a consistent, universal gain from the current predicted-geometry fusion over R18. This does not establish equivalence or prove geometry is generally useless.',
        '- Oracle is separately trained and is neither a guaranteed upper bound nor a causal intervention on M3.',
        '- NME subgroup results describe associations. Group difficulty, attribute prevalence and small subgroup counts can affect comparisons. They do not identify a causal effect of localization error.',
        '- The 1000-replicate paired image-bootstrap F1 intervals are pointwise, unadjusted and conditional on fixed trained models. They exclude training-seed uncertainty. mAP confidence intervals were not computed.',
        '- Seed0 and subsequent runs have recorded environment differences; variability cannot be attributed exclusively to the random seed.',
        '- Twenty-four outcome-stratified qualitative cases illustrate successes, failures and possible annotation ambiguity. They do not estimate label-error prevalence or population model superiority. Dataset labels remain unchanged.',
        '- R18 attribute performance does not demonstrate explicit five-point localization ability.','',
        '## Small-scale reproduction','',
        f'- Seed1, four models x eight fixed test images x four conditions: 128 model-sample evaluations; all replay comparisons passed. Maximum absolute logit difference: {max_error:.9g}.',
        '- Both threshold rules produced zero changed decisions. Reference GPU and replay CPU/hardware details are in the replay execution binding.',
        '- Detector inputs were reused; this is not a fresh MTCNN execution or a full-benchmark replay.',
        '- Runtime is recorded in the replay report. Monetary cost was not automatically measured and is not treated as zero.','',
        '## Evidence index','']
    for title,relative in assets:lines.append(f'- [{title}](../../results/benchmark/{relative})')
    lines += ['', '## Reusable notebooks','']
    for name in notebooks:lines.append(f'- [{name}](../../../notebooks/{name})')
    lines += ['', '## Known metadata issue','',
        'The original case-review analysis_binding.json conditions field mistakenly contains official test IDs because of a variable-name collision. The four actual reviewed conditions are present in completion_audit.json, condition caches and selected_cases.json. Original files and hashes are preserved; no silent replacement is made.','',
        '## Remaining work','',
        '- Complete the detailed interpretation of minority-attribute results and the full case-review narrative.',
        '- Record actual compute spending if required; the current monetary entry is unmeasured.',
        '- Future reuse of case-selection code should fix the metadata field in a new version/output directory.',
        '- Additional controls, training repetitions, detector reruns and broader datasets remain optional future studies, not completed experiments.',
        '- Full paper writing, related-work sourcing, final paper figure selection and demo implementation are not completed by this release.',
        '- October 26 tasks are not specified in the supplied schedule screenshot.',
        '- Checkpoints and full prediction shards remain on Drive. This index does not package them or make the project executable without those prerequisites.','',
        '## Snapshot provenance','',
        f'- Repository HEAD before this release was assembled: `{git_head}`.',
        '- See evidence_manifest.json for file hashes and pre-existing tracked modifications. The archived experiment snapshots, rather than unreviewed working-tree model edits, define the experiments.',
        '- This command does not commit, push, tag, delete, train or run inference.','']
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    evidence=dict(schema_version=1,repository_head_before_release=git_head,
        preexisting_tracked_modified_paths=sorted(dirty),files=tracked_inputs,
        scope='research evidence index; original payloads remain at their archived paths',
        overall_csv_rows_verified_against_reports=252,case_count=24,replay_passed=True,
        known_case_metadata_issue=True,monetary_cost_measured=False)
    writes={output/'README.md':'\n'.join(lines).encode(),output/'summary_metrics.csv':stream.getvalue().encode(),
            output/'evidence_manifest.json':encoded(evidence)}
    writes[output/'release_audit.json']=encoded(dict(complete=True,
        scope='release assembly and input verification only; not completion of all remaining research tasks',
        generated_files={p.name:sha(d) for p,d in writes.items()},verified_input_files=len(tracked_inputs),
        training=False,inference=False,threshold_fitting=False))
    # Validate all destinations before any new write; never overwrite a differing release.
    for path,data in writes.items():
        if path.exists():require(path.read_bytes()==data,f'Existing release differs; preserved: {path}')
    for path,data in writes.items():
        if path.exists():continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.partial')
        with temporary.open('xb') as f:f.write(data)
        temporary.replace(path)
    print('Archived payload hashes and completed analyses: PASS')
    print('Overall CSV/report reproduction: 252/252 rows PASS')
    print('Case review: 24 cases; small-scale replay: 128 evaluations PASS')
    print('Release files: 4; verified input files:',len(tracked_inputs))
    print('Pre-existing tracked modifications recorded:',len(dirty))
    print('Saved:',output)
    print('Research release v1 local assembly: PASS')
    print('No Git mutation, training, inference or threshold fitting performed')


if __name__=='__main__':
    main()
