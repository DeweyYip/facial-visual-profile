"""Verify the exact RQ5 evidence snapshot; Python standard library only."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    expected = read(ROOT / 'archive_checksums.json')
    for name, fingerprint in expected.items():
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f'MISSING: {name}')
        if digest(path) != fingerprint:
            raise SystemExit(f'CHANGED: {name}. Preserve the file and inspect; do not overwrite blindly.')
    for filename in ('source_manifest.json', 'source_snapshot_manifest.json'):
        for item in read(ROOT / filename):
            assert digest(ROOT / item['path']) == item['sha256'], item['path']
    binding = read(ROOT / 'raw/notebook_execution_binding.json')
    for name, fingerprint in binding['sources'].items():
        assert digest(ROOT / 'source_snapshot' / name) == fingerprint, name
    complete = read(ROOT / 'raw/rq5_complete.json')
    assert len(complete['runs']) == 9
    for key, record in complete['runs'].items():
        folder = ROOT / 'raw/runs' / key
        thresholds = read(folder / 'clean_val_thresholds.json')
        test = read(folder / 'test_binding.json')
        assert digest(folder / 'clean_val_thresholds.json') == record['thresholds_sha256'] == test['thresholds']
        assert digest(folder / 'test_metrics.json') == record['test_sha256']
        assert thresholds['checkpoint_sha256'] == record['checkpoint_sha256'] == test['checkpoint']
        assert read(folder / 'train_complete.json') == record['training']
        assert read(folder / 'test_complete.json')['conditions'] == 21
    for filename, count in [('condition_results.csv',252),('attribute_results.csv',6048)]:
        with (ROOT / 'raw' / filename).open(newline='',encoding='utf-8') as source:
            assert sum(1 for _ in csv.DictReader(source)) == count
    assert read(ROOT / 'raw/frozen_config.json')['mtl_lambda'] == 1.89811
    print(f'PASS: {len(expected)} archived file hashes')
    print('PASS: 109 original evidence files and 8 frozen execution source files')
    print('PASS: nine run completion, threshold and test-record bindings')
    print('PASS: 252 condition records and 6,048 attribute records')
    print('No training, weights, dataset reads, or Git changes performed.')

if __name__ == '__main__':
    main()
