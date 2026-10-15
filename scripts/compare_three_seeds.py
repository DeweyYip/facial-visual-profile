#!/usr/bin/env python3
"""Compare archived overall test metrics; no training, inference or fitting.

Usage: python compare_three_seeds.py --archive results.zip --output comparison
Dependencies: numpy, matplotlib. Output SD uses ddof=1 across training seeds.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ('r18', 'r50', 'm3', 'oracle')
FAMILIES = ('blur', 'brightness', 'rotation', 'occlusion', 'jpeg')
CONDITIONS = ('clean_s0',) + tuple(f'{f}_s{s}' for f in FAMILIES for s in range(1, 5))
METRICS = ('f1_frozen', 'f1_at_0.5', 'map')
LABELS = {'f1_frozen': 'F1 (frozen validation thresholds)', 'f1_at_0.5': 'F1 at 0.5', 'map': 'mAP'}
COLORS = dict(zip(MODELS, ('#0072B2', '#E69F00', '#009E73', '#CC79A7')))
PAIRS = (('m3', 'r18'), ('m3', 'r50'), ('oracle', 'm3'), ('r50', 'r18'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_rows(report, run_names):
    result = []
    for item in report['conditions']:
        c = item['condition']
        label = f"{c['condition']}_s{c['severity']}"
        for run in run_names:
            metrics = item['models'][run]
            frozen, fixed = metrics['frozen_thresholds'], metrics['fixed_0_5']
            require(abs(frozen['map'] - fixed['map']) < 1e-12, 'mAP differs between F1 rules')
            require(frozen['sample_count'] == fixed['sample_count'] == 19962, 'Sample count mismatch')
            model, seed = run.split('_seed')
            result.append(dict(condition=label, run_name=run, seed=int(seed), model=model,
                               samples=19962, f1_frozen=frozen['macro_f1'],
                               **{'f1_at_0.5': fixed['macro_f1'], 'map': frozen['map']}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('three_seed_comparison'))
    args = parser.parse_args()
    payload = args.archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        require(archive.testzip() is None, 'ZIP CRC failed')
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        manifest = json.loads(archive.read('archive_manifest.json'))
        require(set(names) == set(manifest['files']) | {'archive_manifest.json'}, 'Manifest membership differs')
        for name, expected in manifest['files'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, f'File hash failed: {name}')
        combined = list(csv.DictReader(io.StringIO(archive.read('analysis/seed0_rgb_m3_plus_seed1_seed2_metrics.csv').decode())))
        rows = []
        for row in combined:
            rows.append({**row, 'seed': int(row['seed']), 'samples': int(row['samples']),
                         **{m: float(row[m]) for m in METRICS}})
        oracle = json.loads(archive.read('source/seed0_oracle_report.json'))
        report = json.loads(archive.read('multiseed_benchmark_report.json'))
        require(report['full_benchmark_complete'] is True, 'Benchmark incomplete')
        require(report['threshold_fitting_performed'] is False, 'Unexpected threshold fitting')
        require(report['attribute_names'] == oracle['attribute_names'], 'Attribute order mismatch')
        rows.extend(report_rows(oracle, ['oracle_seed0']))
        # Independently cross-check all Seed1/2 CSV rows against the JSON report.
        reference = {(r['condition'], r['run_name']): r for r in report_rows(report, report['runs'])}
        for row in rows:
            if row['seed'] in (1, 2):
                other = reference[(row['condition'], row['run_name'])]
                require(all(abs(row[m] - other[m]) < 1e-12 for m in METRICS), 'CSV/JSON mismatch')

    lookup = {}
    for row in rows:
        key = (row['condition'], row['model'], row['seed'])
        require(key not in lookup, f'Duplicate metric row: {key}')
        require(row['run_name'] == f"{row['model']}_seed{row['seed']}", 'Run name mismatch')
        require(row['samples'] == 19962, 'Sample count mismatch')
        require(all(math.isfinite(row[m]) and 0 <= row[m] <= 1 for m in METRICS), 'Invalid metric')
        lookup[key] = row
    expected = {(c, m, s) for c in CONDITIONS for m in MODELS for s in range(3)}
    require(set(lookup) == expected, 'Expected exactly 21 conditions x 4 models x 3 seeds')
    rows = [lookup[c, m, s] for c in CONDITIONS for m in MODELS for s in range(3)]
    out = args.output
    require(out.resolve() != args.archive.resolve(), 'Output cannot replace input')
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'all_seed_metrics.csv', rows)
    summary, paired, degradation = [], [], []
    for c in CONDITIONS:
        for m in MODELS:
            for metric in METRICS:
                values = [lookup[c, m, s][metric] for s in range(3)]
                summary.append(dict(condition=c, model=m, metric=metric, seeds=3,
                                    mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                                    **{f'seed{s}': values[s] for s in range(3)}))
                drops = [100 * (lookup['clean_s0', m, s][metric] - lookup[c, m, s][metric]) for s in range(3)]
                degradation.append(dict(condition=c, model=m, metric=metric,
                                        mean_drop_pp=float(np.mean(drops)), sd_drop_pp=float(np.std(drops, ddof=1)),
                                        **{f'seed{s}_drop_pp': drops[s] for s in range(3)}))
        for a, b in PAIRS:
            for metric in METRICS:
                delta = [100 * (lookup[c, a, s][metric] - lookup[c, b, s][metric]) for s in range(3)]
                paired.append(dict(condition=c, comparison=f'{a}_minus_{b}', metric=metric,
                                   mean_difference_pp=float(np.mean(delta)), sd_difference_pp=float(np.std(delta, ddof=1)),
                                   positive_seeds=sum(v > 0 for v in delta),
                                   **{f'seed{s}_difference_pp': delta[s] for s in range(3)}))
    write_csv(out / 'mean_sd_metrics.csv', summary)
    write_csv(out / 'paired_model_differences_pp.csv', paired)
    write_csv(out / 'clean_relative_degradation_pp.csv', degradation)
    # Each corrupt condition has equal weight; clean is excluded.
    family_summary = []
    for family in (*FAMILIES, 'all_corruptions'):
        conditions = [c for c in CONDITIONS if c != 'clean_s0' and (family == 'all_corruptions' or c.startswith(family + '_'))]
        for m in MODELS:
            for metric in METRICS:
                per_seed = [np.mean([lookup[c, m, s][metric] for c in conditions]) for s in range(3)]
                family_summary.append(dict(family=family, model=m, metric=metric,
                                           mean=float(np.mean(per_seed)), sd=float(np.std(per_seed, ddof=1)),
                                           **{f'seed{s}': float(per_seed[s]) for s in range(3)}))
    write_csv(out / 'corruption_family_summary.csv', family_summary)
    for metric in METRICS:
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.3), sharey=True)
        for axis, family in zip(axes, FAMILIES):
            conditions = ['clean_s0'] + [f'{family}_s{s}' for s in range(1, 5)]
            for model in MODELS:
                values = np.array([[lookup[c, model, seed][metric] for c in conditions] for seed in range(3)])
                mean, sd = values.mean(0), values.std(0, ddof=1)
                axis.plot(range(5), mean, marker='o', ms=3, label=model.upper(), color=COLORS[model])
                axis.fill_between(range(5), mean-sd, mean+sd, color=COLORS[model], alpha=.13)
            axis.set(title=family.title(), xlabel='Severity (0 = clean)', xticks=range(5), ylim=(0, 1))
            axis.grid(alpha=.2)
        axes[0].set_ylabel(LABELS[metric])
        axes[-1].legend(fontsize=8)
        fig.suptitle('Mean across 3 training seeds; shading = sample SD (not a confidence interval)')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out / f"curves_{metric.replace('.', '_')}.{ext}", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for axis, metric in zip(axes, METRICS):
        for i, seed in enumerate(range(3)):
            values = [100*(lookup[c, 'm3', seed][metric]-lookup[c, 'r18', seed][metric]) for c in CONDITIONS]
            axis.plot(values, range(21), marker='o', ms=3, label=f'Seed {seed}')
        axis.axvline(0, color='black', lw=.8)
        axis.set(title=LABELS[metric], xlabel='M3 - R18 (percentage points)', yticks=range(21), yticklabels=CONDITIONS)
        axis.invert_yaxis()
        axis.grid(alpha=.2)
    axes[-1].legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'paired_m3_r18.{ext}', dpi=180)
    plt.close(fig)
    notes = ['# Three-seed overall comparison', '',
             'Verified 252 overall rows: 21 conditions x 4 models x 3 training seeds.',
             'F1(frozen), F1@0.5 and mAP are reported separately. SD uses ddof=1.',
             'Model differences pair the same seed and condition. Values in difference tables are percentage points.',
             'Family summaries average severity levels within each seed, then summarize the three seed averages.',
             'These are descriptive results. No significance tests or confidence intervals were computed.',
             'Test-image uncertainty, NME subgroup analysis and prediction-shard revalidation are not included.',
             'Oracle is an independently trained GT-reference model, not a guaranteed upper bound.',
             'Seed0 and later seeds have recorded environment differences; do not attribute all variation solely to random seed.', '',
             '## Rotation severity 4: M3 minus R18', '']
    for row in paired:
        if row['condition'] == 'rotation_s4' and row['comparison'] == 'm3_minus_r18':
            values = ', '.join(f"Seed{s}: {row[f'seed{s}_difference_pp']:+.3f} pp" for s in range(3))
            notes.append(f"- {row['metric']}: {values}; mean {row['mean_difference_pp']:+.3f} pp, SD {row['sd_difference_pp']:.3f} pp.")
    (out / 'README.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    audit = dict(archive_sha256=hashlib.sha256(payload).hexdigest(), archive_member_hashes_verified=len(manifest['files']),
                 overall_rows=len(rows), seed1_seed2_csv_json_crosscheck=True, seeds=[0,1,2],
                 training_performed=False, inference_performed=False, threshold_fitting_performed=False,
                 significance_analysis_performed=False)
    (out / 'comparison_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print('\n'.join(notes))
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
