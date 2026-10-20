#!/usr/bin/env python3
"""Plot seeds 0-3 and a separately labeled seed-0 remake from frozen reports.

Run: python3 scripts/compare_four_seeds_and_remake.py --project .
Dependencies: numpy, matplotlib. No model inference or threshold fitting.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ('r18', 'r50', 'm3', 'oracle')
FAMILIES = ('blur', 'brightness', 'rotation', 'occlusion', 'jpeg')
CONDITIONS = ('clean_s0',) + tuple(f'{f}_s{s}' for f in FAMILIES for s in range(1, 5))
VARIANTS = ('seed0', 'seed1', 'seed2', 'seed3', 'seed0_remake')
INDEPENDENT = VARIANTS[:4]
METRICS = ('f1_frozen', 'f1_at_0_5', 'map')
LABELS = {'f1_frozen': 'F1 (frozen validation thresholds)', 'f1_at_0_5': 'F1 at 0.5', 'map': 'mAP'}
ATTR_KEYS = {'f1_frozen': 'f1_frozen', 'f1_at_0_5': 'f1_at_0_5', 'map': 'ap'}
MODEL_COLORS = dict(zip(MODELS, ('#0072B2', '#E69F00', '#009E73', '#CC79A7')))
RUN_COLORS = dict(zip(VARIANTS, ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#333333')))
RUN_LABELS = {'seed0': 'Seed 0 (original)', 'seed1': 'Seed 1', 'seed2': 'Seed 2', 'seed3': 'Seed 3', 'seed0_remake': 'Seed 0 remake'}
PAIRS = (('m3', 'r18'), ('m3', 'r50'), ('oracle', 'm3'), ('r50', 'r18'))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_csv(path, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding='utf-8')


def read_inputs(project):
    base = project / 'docs/results/benchmark'
    hashes = {}

    def read(path, expected=None):
        raw = path.read_bytes()
        digest = sha(raw)
        if expected:
            require(digest == expected, f'Source hash mismatch: {path}')
        hashes[path.relative_to(project).as_posix()] = digest
        return json.loads(raw)

    def archived(folder, filename):
        root = base / folder
        manifest = read(root / 'archive_manifest.json')
        report = read(root / filename, manifest['files'][filename])
        binding = read(root / 'execution_binding.json', report['execution_binding_sha256'])
        require(report['full_benchmark_complete'] is True, 'Incomplete benchmark')
        require(report['test_evaluated'] is True, 'Missing test evaluation')
        require(report['threshold_fitting_performed'] is False, 'Test threshold fitting detected')
        return report, binding, manifest

    old, old_binding, old_manifest = archived('multiseed_seed1_seed2', 'multiseed_benchmark_report.json')
    new, _, _ = archived('seed0_reproduction_v1_seed3', 'multiseed_benchmark_report.json')
    require(old['seeds'] == [1, 2] and new['seeds'] == [0, 3], 'Unexpected source seed sets')
    require(new['run_tags'] == {'0': 'seed0_reproduction_v1', '3': 'seed3'}, 'Incorrect remake mapping')
    original_reports = []
    for day in ('day19', 'day20'):
        filename = f'{day}_benchmark_report.json'
        report = read(base / day / filename, old_binding['source_reports'][day][filename])
        read(base / day / 'execution_binding.json', report['execution_binding_sha256'])
        original_reports.append(report)
    oracle = read(base / 'multiseed_seed1_seed2/source/seed0_oracle_report.json', old_manifest['files']['source/seed0_oracle_report.json'])
    attributes = old['attribute_names']
    require(len(attributes) == len(set(attributes)) == 24, 'Invalid attribute order')
    lookup, attribute_lookup, thresholds = {}, {}, {}
    positive_counts = None
    for report in (*original_reports, oracle, old, new):
        require(report['attribute_names'] == attributes, 'Attribute order mismatch')
        require(report['threshold_fitting_performed'] is False, 'Unexpected threshold fitting')
        for item in report['conditions']:
            condition = item['condition']
            label = f"{condition['condition']}_s{condition['severity']}"
            require(label in CONDITIONS, f'Unexpected condition: {label}')
            for run, values in item['models'].items():
                model, suffix = run.split('_seed', 1)
                variant = 'seed0_remake' if suffix == '0_reproduction_v1' else f'seed{suffix}'
                require(model in MODELS and variant in VARIANTS, f'Unexpected run: {run}')
                frozen, fixed = values['frozen_thresholds'], values['fixed_0_5']
                require(frozen['sample_count'] == fixed['sample_count'] == 19962, 'Sample count mismatch')
                require(fixed['thresholds'] == [0.5] * 24, 'Fixed-threshold rule differs')
                require(frozen['per_attribute_ap'] == fixed['per_attribute_ap'], 'AP changed with threshold rule')
                if run in thresholds:
                    require(thresholds[run] == frozen['thresholds'], 'Thresholds changed between conditions')
                thresholds[run] = frozen['thresholds']
                counts = frozen['positive_counts']
                require(counts == fixed['positive_counts'], 'Target counts differ')
                if positive_counts is not None:
                    require(positive_counts == counts, 'Target counts differ between runs/conditions')
                positive_counts = counts
                arrays = (frozen['per_attribute_f1'], fixed['per_attribute_f1'], frozen['per_attribute_ap'])
                for metric, array in zip(METRICS, arrays):
                    require(len(array) == 24 and np.isfinite(array).all() and min(array) >= 0 and max(array) <= 1, 'Invalid attribute metrics')
                    expected = frozen['map'] if metric == 'map' else frozen['macro_f1'] if metric == 'f1_frozen' else fixed['macro_f1']
                    require(abs(float(np.mean(array)) - expected) <= 1e-12, 'Macro metric reconstruction failed')
                key = (label, model, variant)
                require(key not in lookup, f'Duplicate run-condition: {key}')
                lookup[key] = dict(condition=label, model=model, variant=variant, nominal_seed=int(suffix.split('_')[0]), independent_seed=variant in INDEPENDENT, run_name=run, samples=19962, f1_frozen=frozen['macro_f1'], f1_at_0_5=fixed['macro_f1'], map=frozen['map'])
                for index, attribute in enumerate(attributes):
                    attribute_lookup[label, model, variant, attribute] = dict(condition=label, model=model, variant=variant, run_name=run, attribute=attribute, samples=19962, positives=counts[index], threshold=frozen['thresholds'][index], f1_frozen=arrays[0][index], f1_at_0_5=arrays[1][index], ap=arrays[2][index])
    expected = {(c, m, v) for c in CONDITIONS for m in MODELS for v in VARIANTS}
    require(set(lookup) == expected, 'Expected 420 overall rows')
    require(len(attribute_lookup) == 10080, 'Expected 10,080 attribute rows')
    return attributes, lookup, attribute_lookup, hashes


def save_figure(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_outputs(project, out):
    attributes, data, adata, hashes = read_inputs(project)
    require(not out.exists() or not any(out.iterdir()), 'Choose an empty output directory to preserve prior figures')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    save_csv(out / 'all_five_run_metrics.csv', [data[c, m, v] for c in CONDITIONS for m in MODELS for v in VARIANTS])
    save_csv(out / 'all_five_run_attribute_metrics.csv', [adata[c, m, v, a] for c in CONDITIONS for m in MODELS for v in VARIANTS for a in attributes])
    mean_rows, paired, paired_raw, degradation, family_rows, remake_rows = [], [], [], [], [], []
    for c in CONDITIONS:
        for m in MODELS:
            for metric in METRICS:
                vals = [data[c, m, v][metric] for v in INDEPENDENT]
                mean_rows.append(dict(condition=c, model=m, metric=metric, independent_seeds=4, mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), **dict(zip(INDEPENDENT, vals))))
                for v in VARIANTS:
                    degradation.append(dict(condition=c, model=m, variant=v, metric=metric, clean_relative_drop_pp=100 * (data['clean_s0', m, v][metric] - data[c, m, v][metric])))
                original, remake = data[c, m, 'seed0'][metric], data[c, m, 'seed0_remake'][metric]
                remake_rows.append(dict(condition=c, model=m, metric=metric, original=original, remake=remake, remake_minus_original_pp=100 * (remake - original)))
        for lhs, rhs in PAIRS:
            for metric in METRICS:
                values = [100 * (data[c, lhs, v][metric] - data[c, rhs, v][metric]) for v in VARIANTS]
                paired.append(dict(condition=c, comparison=f'{lhs}_minus_{rhs}', metric=metric, independent_seeds=4, mean_difference_pp=float(np.mean(values[:4])), sd_difference_pp=float(np.std(values[:4], ddof=1)), positive_seeds=sum(x > 0 for x in values[:4]), **{f'{v}_difference_pp': value for v, value in zip(VARIANTS, values)}))
                for v, value in zip(VARIANTS, values):
                    paired_raw.append(dict(condition=c, comparison=f'{lhs}_minus_{rhs}', variant=v, metric=metric, difference_pp=value))
    for family in (*FAMILIES, 'all_corruptions'):
        cs = [c for c in CONDITIONS[1:] if family == 'all_corruptions' or c.startswith(family + '_')]
        for m in MODELS:
            for metric in METRICS:
                vals = [float(np.mean([data[c, m, v][metric] for c in cs])) for v in VARIANTS]
                family_rows.append(dict(family=family, model=m, metric=metric, independent_seeds=4, mean=float(np.mean(vals[:4])), sd=float(np.std(vals[:4], ddof=1)), **dict(zip(VARIANTS, vals))))
    for name, rows in [('four_seed_mean_sd.csv', mean_rows), ('paired_model_differences_pp.csv', paired), ('all_five_run_paired_differences_pp.csv', paired_raw), ('clean_relative_degradation_pp.csv', degradation), ('corruption_family_summary.csv', family_rows), ('seed0_remake_differences_pp.csv', remake_rows)]:
        save_csv(out / name, rows)
    for metric in METRICS:
        fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharey=True)
        for ax, family in zip(axes, FAMILIES):
            cs = ['clean_s0'] + [f'{family}_s{i}' for i in range(1, 5)]
            for model in MODELS:
                vals = np.array([[data[c, model, v][metric] for c in cs] for v in INDEPENDENT])
                mean, sd = vals.mean(0), vals.std(0, ddof=1)
                ax.plot(range(5), mean, marker='o', ms=3, color=MODEL_COLORS[model], label=model.upper())
                ax.fill_between(range(5), mean-sd, mean+sd, alpha=.13, color=MODEL_COLORS[model])
            ax.set(title=family.title(), xlabel='Severity (0 = clean)', xticks=range(5), ylim=(0, 1))
            ax.grid(alpha=.18)
        axes[0].set_ylabel(LABELS[metric]);axes[-1].legend()
        fig.suptitle('Seeds 0-3: mean and sample SD | Seed 0 remake excluded from aggregation', fontsize=12)
        fig.tight_layout();save_figure(fig, out, f'four_seed_curves_{metric}')
        # Show every model and run separately, using consistent run colors.
        fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharex=True, sharey=True)
        for row, model in enumerate(MODELS):
            for column, family in enumerate(FAMILIES):
                ax = axes[row, column];cs = ['clean_s0'] + [f'{family}_s{i}' for i in range(1, 5)]
                for variant in VARIANTS:
                    ax.plot(range(5), [data[c, model, variant][metric] for c in cs], color=RUN_COLORS[variant], ls='--' if variant == 'seed0_remake' else '-', marker='x' if variant == 'seed0_remake' else 'o', ms=3, lw=1.2, label=RUN_LABELS[variant])
                ax.set(xticks=range(5), ylim=(0, 1));ax.grid(alpha=.18)
                if row == 0:ax.set_title(family.title())
                if column == 0:ax.set_ylabel(model.upper() + '\n' + LABELS[metric])
                if row == 3:ax.set_xlabel('Severity (0 = clean)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .965), ncol=5, frameon=False)
        fig.suptitle('All runs | Dashed black = seed 0 remake (a repeat, not a fifth seed)', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .93));save_figure(fig, out, f'all_five_run_curves_{metric}')
    # Retain the original condition-by-condition comparison format.
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    for ax, metric in zip(axes, METRICS):
        for variant in VARIANTS:
            vals = [100*(data[c, 'm3', variant][metric]-data[c, 'r18', variant][metric]) for c in CONDITIONS]
            ax.plot(vals, range(21), marker='x' if variant == 'seed0_remake' else 'o', ms=4, lw=1.3, ls='--' if variant == 'seed0_remake' else '-', color=RUN_COLORS[variant], label=RUN_LABELS[variant])
        ax.axvline(0, color='#555555', lw=.8);ax.grid(alpha=.18)
        ax.set(title=LABELS[metric], xlabel='M3 - R18 (percentage points)', yticks=range(21), yticklabels=CONDITIONS)
        ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=5, frameon=False)
    fig.suptitle('M3 versus R18: four original seeds and a separate seed-0 reproduction', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .91));save_figure(fig, out, 'paired_m3_r18_all_five_runs')
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    for ax, metric in zip(axes, METRICS):
        for model in MODELS:
            vals = [100*(data[c, model, 'seed0_remake'][metric]-data[c, model, 'seed0'][metric]) for c in CONDITIONS]
            ax.plot(vals, range(21), marker='o', ms=3, color=MODEL_COLORS[model], label=model.upper())
        ax.axvline(0, color='#555555', lw=.8);ax.grid(alpha=.18)
        ax.set(title=LABELS[metric], xlabel='Remake - original seed 0 (percentage points)', yticks=range(21), yticklabels=CONDITIONS)
        ax.invert_yaxis()
    axes[-1].legend();fig.suptitle('Seed 0 reproduction: changes in each model, with its own frozen thresholds', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95));save_figure(fig, out, 'seed0_remake_vs_original')
    # Attribute heatmaps average four original seeds, with every attribute retained.
    attribute_deltas = []
    groups = ('clean', *FAMILIES, 'all_corruptions')
    for group in groups:
        cs = ['clean_s0'] if group == 'clean' else list(CONDITIONS[1:]) if group == 'all_corruptions' else [f'{group}_s{i}' for i in range(1, 5)]
        for lhs, rhs in (('m3', 'r18'), ('oracle', 'm3')):
            for metric in METRICS:
                am = ATTR_KEYS[metric]
                for a in attributes:
                    vals = [100*float(np.mean([adata[c, lhs, v, a][am]-adata[c, rhs, v, a][am] for c in cs])) for v in VARIANTS]
                    attribute_deltas.append(dict(family=group, comparison=f'{lhs}_minus_{rhs}', attribute=a, metric='ap' if metric == 'map' else metric, independent_seeds=4, mean_difference_pp=float(np.mean(vals[:4])), sd_difference_pp=float(np.std(vals[:4], ddof=1)), **{f'{v}_difference_pp': val for v, val in zip(VARIANTS, vals)}))
    save_csv(out / 'attribute_family_differences_pp.csv', attribute_deltas)
    selected = groups[:6]
    for lhs, rhs in (('m3', 'r18'), ('oracle', 'm3')):
        matrices = [np.array([[next(r['mean_difference_pp'] for r in attribute_deltas if r['family']==g and r['comparison']==f'{lhs}_minus_{rhs}' and r['attribute']==a and r['metric']==('ap' if metric=='map' else metric)) for g in selected] for a in attributes]) for metric in METRICS]
        limit = max(1., max(float(np.abs(v).max()) for v in matrices))
        fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
        for ax, metric, matrix in zip(axes, METRICS, matrices):
            im=ax.imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            ax.set(title='AP' if metric=='map' else LABELS[metric], xticks=range(6), xticklabels=selected, yticks=range(24), yticklabels=attributes)
            ax.tick_params(axis='x', rotation=40)
        fig.colorbar(im, ax=axes.tolist(), label='Difference (percentage points)', fraction=.025, pad=.025)
        fig.suptitle(f'{lhs.upper()} - {rhs.upper()}: mean across seeds 0-3\nFour severity levels averaged within each corruption family; remake excluded', fontsize=12)
        fig.subplots_adjust(left=.15, right=.88, bottom=.12, top=.9, wspace=.15)
        save_figure(fig, out, f'attribute_heatmap_{lhs}_minus_{rhs}')
    max_m3 = max(abs(r['remake_minus_original_pp']) for r in remake_rows if r['model']=='m3')
    notes = ['# Four original seeds and seed-0 remake comparison', '',
        'Coverage: 420 overall rows and 10,080 attribute rows = 21 conditions x 4 models x 5 labeled runs.',
        'Independent nominal seeds: original seed0, seed1, seed2 and seed3. Seed0 remake is a repeated run and excluded from every mean/SD calculation.',
        'No result is removed or replaced. Original seed0 remains included in the four-seed descriptive summaries.',
        'SD uses ddof=1; it is not a confidence interval. Recorded historical environment differences remain a confound, so variation is not attributed solely to seed.',
        'F1(frozen), F1@0.5 and mAP/AP are separate. Each run uses its own previously frozen validation thresholds.',
        'Family summaries weight severities equally within each run before summarizing the four original seeds.',
        'Five-run plots use dashed black for the seed0 remake. The legend and CSV retain it even when curves overlap.',
        'Paired model differences match condition and nominal run. Seed0 reproduction differences compare remake minus original for each model.',
        f'Maximum absolute M3 original/remake aggregate F1/mAP difference: {max_m3:.12g} percentage points. This does not imply identical weights or individual logits.',
        'Oracle is independently trained and is not a guaranteed upper bound. Oracle minus M3 is not a causal intervention on detector accuracy.',
        'No training, inference, threshold fitting, shard rereading, bootstrap intervals or significance tests performed.',
        '', '## Figure guide', '',
        '- paired_m3_r18_all_five_runs: direct replacement for the old three-seed comparison.',
        '- all_five_run_curves_*: every model and run shown separately.',
        '- four_seed_curves_*: four-original-seed mean with sample-SD shading.',
        '- seed0_remake_vs_original: remake minus original, per model and condition.',
        '- attribute_heatmap_*: all 24 attributes, four-original-seed mean.',
        '', '## Reproduce', '',
        'python3 scripts/compare_four_seeds_and_remake.py --project . --output new_empty_comparison_folder', '']
    (out / 'README.md').write_text('\n'.join(notes), encoding='utf-8')
    audit = dict(complete=True, independent_seeds=[0,1,2,3], remake_variant='seed0_remake', remake_in_mean_sd=False, overall_rows=len(data), attribute_rows=len(adata), sample_count=19962, conditions=21, source_sha256=hashes, per_attribute_macro_reconstruction=True, positive_counts_aligned=True, frozen_thresholds_constant=True, training=False, inference=False, threshold_fitting=False, statistical_confidence_intervals=False, max_m3_remake_difference_pp=max_m3)
    (out / 'comparison_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('PASS: 420 overall rows; 10,080 attribute rows; hashes and macro reconstruction checked.')
    print('Saved:', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    out = args.output or project / 'docs/results/benchmark/four_seed_and_remake_comparison'
    make_outputs(project, out.resolve())


if __name__ == '__main__':
    main()
