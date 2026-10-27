"""RQ5 inputs using the RQ4 canvas and existing M3 cache readers."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from datasets.landmark_cache import MTCNNLandmarkCache
from training.rq5_shared import prepare_canvas, transform_corruption, normalize_geometry, MEAN, STD

ATTRIBUTES = [
    'Smiling', 'Mouth_Slightly_Open', 'Eyeglasses', 'Bangs', 'Wearing_Hat',
    'Mustache', 'No_Beard', 'Black_Hair', 'Bushy_Eyebrows', 'Arched_Eyebrows',
    'High_Cheekbones', 'Bags_Under_Eyes', 'Big_Lips', 'Big_Nose', 'Double_Chin',
    'Oval_Face', 'Receding_Hairline', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
    'Bald', 'Straight_Hair', 'Wavy_Hair', 'Sideburns'
]


def read_annotation(path, columns):
    with open(path, encoding='utf-8-sig') as source:
        count = int(next(source))
        header = next(source).split()
        indices = [header.index(key) for key in columns]
        rows = {}
        for line in source:
            values = line.split()
            if values[0] in rows:
                raise ValueError('Duplicate annotation filename')
            rows[values[0]] = np.array([float(values[1+i]) for i in indices], dtype=np.float32)
    if len(rows) != count:
        raise ValueError('Annotation count mismatch')
    return rows


class BenchmarkDetections:
    """Read original benchmark shard records, retaining RGB hash provenance.

    Supply the old day19/shards and day20/shards directories. ZIP manifests
    are verified; detector results are reused, never recomputed or guessed.
    """

    def __init__(self, roots, family, severity):
        self.records = {}
        label = f'{family or "clean"}_s{severity}'
        for root in roots:
            for path in sorted((Path(root) / label).glob('*.zip')):
                with zipfile.ZipFile(path) as archive:
                    manifest = json.loads(archive.read('manifest.json'))
                    for name, expected in manifest['files'].items():
                        payload = archive.read(name)
                        # Existing shard manifests bind payload bytes by SHA256.
                        actual = hashlib.sha256(payload).hexdigest()
                        target = expected if isinstance(expected, str) else expected['sha256']
                        if actual != target:
                            raise ValueError(f'Shard payload changed: {path}/{name}')
                    rows = json.loads(archive.read('records.json'))
                for record in rows:
                    key = record['sample_id']
                    if key in self.records:
                        raise ValueError(f'Duplicate benchmark detection: {key}')
                    self.records[key] = record
        if not self.records:
            raise FileNotFoundError(f'No original benchmark records for {label}')


class CelebARQ5(Dataset):
    """Return image, attributes, predicted coordinates/mask and GT target.

    Images must already be extracted. This class never modifies source data.
    N0/MTL do not open any detector cache.
    """

    def __init__(self, project, image_root, split, model, family=None, severity=0,
                 benchmark_roots=()):
        self.project = Path(project)
        self.image_root = Path(image_root)
        self.model, self.family, self.severity = model, family, severity
        raw = self.project / 'data'
        if not (raw / 'list_eval_partition.txt').is_file():
            raw = raw / 'raw'
        pairs = [line.split() for line in (raw/'list_eval_partition.txt').read_text().splitlines()]
        codes = {'train': '0', 'valid': '1', 'test': '2'}
        if len(dict(pairs)) != 202599 or len(pairs) != 202599:
            raise ValueError('Official partition count/uniqueness mismatch')
        for key, count in [('train',162770),('valid',19867),('test',19962)]:
            if sum(code == codes[key] for _,code in pairs) != count:
                raise ValueError('Official split counts differ')
        self.names = sorted(name for name,code in pairs if code == codes[split])
        self.targets = read_annotation(raw/'list_attr_celeba.txt', ATTRIBUTES)
        self.landmarks = read_annotation(raw/'list_landmarks_align_celeba.txt',
            ['lefteye_x','lefteye_y','righteye_x','righteye_y','nose_x','nose_y',
             'leftmouth_x','leftmouth_y','rightmouth_x','rightmouth_y'])
        if set(self.targets) != set(dict(pairs)) or set(self.landmarks) != set(dict(pairs)):
            raise ValueError('Annotation and partition IDs differ')
        for name in self.names:
            y = self.targets[name]
            if not np.isin(y, [-1, 1]).all():
                raise ValueError('Invalid attribute label')
            self.targets[name] = (y + 1) / 2
        if split != 'test' and (family is not None or severity != 0):
            raise ValueError('RQ5 training/validation must remain clean')
        self.cache = None
        self.benchmark = None
        if model == 'N1':
            if split == 'test':
                self.benchmark = BenchmarkDetections(benchmark_roots, family, severity)
                if set(self.benchmark.records) != set(self.names):
                    raise ValueError('Benchmark cache must cover all test images')
            else:
                self.cache = MTCNNLandmarkCache(self.project/'cache/mtcnn_clean.sqlite',
                                                raw/'list_eval_partition.txt', split)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.image_root/name) as image:
            canvas, gt = prepare_canvas(image, self.landmarks[name])
            raw_points, status = (self.cache.get(name, image.size) if self.cache else (None, 'unused'))
            if raw_points is not None:
                _, predicted = prepare_canvas(image, raw_points)
        canvas, gt = transform_corruption(canvas, gt, self.family, self.severity, name)
        points = torch.zeros(10, dtype=torch.float32)
        valid = False
        if self.benchmark:
            record = self.benchmark.records[name]
            # Same RGB hash definition as datasets/corruptions.py must be audited.
            from datasets.corruptions import rgb_sha256
            if rgb_sha256(canvas) != record['final_rgb_sha256']:
                raise ValueError(f'Frozen test RGB differs: {name}; stop, do not regenerate cache')
            detection = record['detection']
            status = detection['status']
            if status not in ('ok','no_face','invalid_output'):
                raise ValueError(f'Unknown detection status: {status}')
            valid = status == 'ok'
            if valid:
                points = normalize_geometry(detection['landmarks'])
        elif raw_points is not None:
            valid = True
            points = normalize_geometry(predicted)
        if not torch.isfinite(points).all():
            raise ValueError(f'Invalid predicted coordinates: {name}')
        return {'image': TF.normalize(TF.to_tensor(canvas), MEAN, STD),
                'target': torch.from_numpy(self.targets[name].copy()),
                'points': points, 'valid': torch.tensor(valid),
                'gt': normalize_geometry(gt), 'name': name}
