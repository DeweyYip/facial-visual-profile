"""CPU synthetic checks. Does not read CelebA or run any training/evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, __version__ as pillow_version, features
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.preprocessing import prepare
from datasets.benchmark_preprocessing import prepare_canvas, normalize_canvas, normalize_points
from datasets.corruptions import LEVELS, VERSION, apply_corruption

config = json.loads((ROOT / "configs/corruptions_draft_v1.json").read_text())
assert config["status"] == "validation_preview_draft_not_frozen"
assert config["implementation_version"] == VERSION
assert config["levels"] == {key: list(value) for key, value in LEVELS.items()}
assert config["test_access"] is False
assert config["corruption_seed"] == 0

rng = np.random.default_rng(14)
raw = Image.fromarray(rng.integers(0, 256, (218, 178, 3), dtype=np.uint8))
landmarks = torch.tensor([69,109,106,113,77,142,73,152,108,154])
original_tensor, original_points = prepare(raw, landmarks)
canvas, pixels = prepare_canvas(raw, landmarks)
assert torch.equal(normalize_canvas(canvas), original_tensor)
assert torch.equal(normalize_points(pixels), original_points)
print("Clean preprocessing equivalence: PASS (exact tensor equality)")

results = {}
for kind in LEVELS:
    checks = []
    for severity in range(5):
        kwargs = dict(sample_id="synthetic.jpg", seed=0, gt_pixels=pixels.numpy())
        image, points, meta = apply_corruption(canvas, kind, severity, **kwargs)
        duplicate, points2, meta2 = apply_corruption(canvas, kind, severity, **kwargs)
        assert image.tobytes() == duplicate.tobytes() and meta == meta2
        assert np.array_equal(points, points2)
        assert image.mode == "RGB" and image.size == (224, 224)
        tensor = normalize_canvas(image)
        assert tensor.shape == (3, 224, 224) and torch.isfinite(tensor).all()
        if severity == 0:
            assert torch.equal(tensor, original_tensor)
        if kind != "rotation" or severity == 0:
            assert np.array_equal(points, pixels.numpy())
        if kind == "occlusion" and severity:
            x0, y0, x1, y1 = meta["parameters"]["box_xyxy_exclusive"]
            assert not np.asarray(image)[y0:y1, x0:x1].any()
        checks.append(meta)
    results[kind] = checks
    print(f"{kind}: severity 0..4; deterministic RGB, finite tensor, GT checks: PASS")

# Independent image/point agreement check using bright point markers.
marker_points = np.array([[55.,65.],[164.,65.],[111.,111.],[75.,165.],[148.,165.]])
marker_array = np.zeros((224,224,3), dtype=np.uint8)
for x, y in marker_points.astype(int):
    marker_array[y-1:y+2, x-1:x+2] = 255
marker_canvas = Image.fromarray(marker_array)
max_marker_error = 0.0
signs = set()
for sample_id in [f"marker_{i}.jpg" for i in range(32)]:
    sign_for_sample = set()
    for severity in range(1,5):
        rotated, transformed, meta = apply_corruption(
            marker_canvas, "rotation", severity, sample_id=sample_id,
            gt_pixels=marker_points)
        sign = int(np.sign(meta["parameters"]["angle_degrees_ccw"]))
        signs.add(sign)
        sign_for_sample.add(sign)
        intensity = np.asarray(rotated, dtype=np.float64).mean(axis=2)
        for x,y in transformed:
            xi,yi = int(round(x)),int(round(y))
            patch = intensity[yi-5:yi+6,xi-5:xi+6]
            yy,xx = np.mgrid[yi-5:yi+6,xi-5:xi+6]
            assert patch.sum() > 0
            error = float(np.hypot((xx*patch).sum()/patch.sum()-x,
                                   (yy*patch).sum()/patch.sum()-y))
            max_marker_error = max(max_marker_error,error)
    assert len(sign_for_sample) == 1
assert signs == {-1,1}
assert max_marker_error < 0.5
print(f"Rotation image/GT marker alignment, both directions: PASS ({max_marker_error:.6f}px)")

report = {
    "scope": "synthetic CPU smoke checks; not real-validation preview or test evaluation",
    "implementation_version": VERSION, "draft_not_frozen": True,
    "pillow": pillow_version, "libjpeg": features.version_codec("jpg"),
    "numpy": np.__version__, "torch": str(torch.__version__),
    "clean_equivalence_exact": True, "rotation_marker_max_error_px": max_marker_error,
    "synthetic_records": results,
    "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "datasets/preprocessing.py", ROOT / "datasets/benchmark_preprocessing.py",
                     ROOT / "datasets/corruptions.py", ROOT / "configs/corruptions_draft_v1.json"]},
    "test_evaluated": False, "passed": True,
}
destination = ROOT / "runs/corruption_preparation/day14/synthetic_smoke_report.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print("Saved:", destination)
print("Day 14 synthetic corruption smoke: PASS")
