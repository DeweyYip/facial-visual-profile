"""Audit and visualize GT landmarks on Day 14 validation images."""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.corruptions import apply_corruption, rgb_sha256

preview_dir = ROOT / "runs/corruption_preparation/day14/validation_preview"
report_path = (
    ROOT / "docs/results/corruption_preparation/day14"
    / "validation_preview_report.json"
)
output_dir = ROOT / "runs/corruption_preparation/day15"

report = json.loads(report_path.read_text(encoding="utf-8"))
assert report["passed"] is True
assert report["split"] == "validation"
assert report["test_evaluated"] is False

import ast
import subprocess

class RemoveDocumentation(ast.NodeTransformer):
    def visit_Expr(self, node):
        # Ignore standalone string explanations and docstrings.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

def computational_ast(content):
    tree = ast.parse(content.decode("utf-8"))
    tree = RemoveDocumentation().visit(tree)
    return ast.dump(tree, include_attributes=False)

source_verification = {}
for filename, expected in report["source_sha256"].items():
    current = (ROOT / filename).read_bytes()
    actual = hashlib.sha256(current).hexdigest()
    status = "exact_hash_match"

    if actual != expected:
        assert filename.endswith(".py"), f"Configuration changed: {filename}"
        original = subprocess.check_output(
            ["git", "show", f"ed0451e:{filename}"], cwd=ROOT
        )
        assert hashlib.sha256(original).hexdigest() == expected, (
            f"Day 14 commit does not match recorded source: {filename}"
        )
        assert computational_ast(current) == computational_ast(original), (
            f"Computational code changed: {filename}"
        )
        status = "documentation_only_difference"
        print(f"{filename}: documentation changed; computational AST identical")

    source_verification[filename] = {
        "day14_recorded_sha256": expected,
        "current_sha256": actual,
        "verification": status,
    }

names = report["sample_filenames"]
kinds = ("blur", "brightness", "rotation", "occlusion", "jpeg")
records = {
    (row["sample_id"], row["kind"], row["severity"]): row
    for row in report["records"]
}
assert len(records) == len(report["records"]) == 100
assert set(records) == {
    (name, kind, severity)
    for name in names
    for kind in kinds
    for severity in range(5)
}

output_dir.mkdir(parents=True, exist_ok=True)

# Indices preserve the dataset's eye/mouth point order.
colors = ("cyan", "yellow", "lime", "magenta", "orange")
audits = []
maximum_coordinate_error = 0.0
rotation_signs = set()

for name in names:
    clean_record = records[(name, "blur", 0)]
    clean_path = preview_dir / clean_record["relative_png"]
    with Image.open(clean_path) as image:
        clean = image.convert("RGB")

    clean_gt = np.asarray(clean_record["gt_pixels"], dtype=np.float64)
    assert clean_gt.shape == (5, 2)
    assert np.isfinite(clean_gt).all()
    assert rgb_sha256(clean) == clean_record["output_rgb_sha256"]

    montage = Image.new("RGB", (5 * 224, 5 * 250), "white")
    headings = ImageDraw.Draw(montage)

    for row, kind in enumerate(kinds):
        for severity in range(5):
            record = records[(name, kind, severity)]
            saved_gt = np.asarray(record["gt_pixels"], dtype=np.float64)

            with Image.open(preview_dir / record["relative_png"]) as image:
                final = image.convert("RGB")
            assert rgb_sha256(final) == record["output_rgb_sha256"]
            assert rgb_sha256(clean) == record["input_rgb_sha256"]

            # Recreate from clean, never from a previous severity.
            rebuilt, rebuilt_gt, rebuilt_metadata = apply_corruption(
                clean, kind, severity,
                sample_id=name,
                seed=report["corruption_seed"],
                gt_pixels=clean_gt,
            )
            assert rebuilt.tobytes() == final.tobytes()
            assert rebuilt_metadata["parameters"] == record["parameters"]
            assert np.allclose(rebuilt_gt, saved_gt, atol=1e-8, rtol=0)

            # Independently compute expected GT coordinates.
            expected_gt = clean_gt.copy()
            if kind == "rotation" and severity:
                angle = record["parameters"]["angle_degrees_ccw"]
                rotation_signs.add(1 if angle > 0 else -1)
                theta = math.radians(angle)
                dx = clean_gt[:, 0] - 111.5
                dy = clean_gt[:, 1] - 111.5
                expected_gt[:, 0] = (
                    math.cos(theta) * dx + math.sin(theta) * dy + 111.5
                )
                expected_gt[:, 1] = (
                    -math.sin(theta) * dx + math.cos(theta) * dy + 111.5
                )

            error = float(np.max(np.abs(saved_gt - expected_gt)))
            assert error < 1e-8
            maximum_coordinate_error = max(maximum_coordinate_error, error)

            in_frame = (
                (saved_gt >= 0).all(axis=1)
                & (saved_gt <= 223).all(axis=1)
            )
            assert in_frame.tolist() == record["gt_in_frame"]

            # Draw only on a copy; model input pixels remain unchanged.
            overlay = final.copy()
            draw = ImageDraw.Draw(overlay)
            for index, ((x, y), visible) in enumerate(zip(saved_gt, in_frame)):
                if visible:
                    draw.ellipse(
                        (x - 3, y - 3, x + 3, y + 3),
                        fill=colors[index], outline="black",
                    )
                    draw.text((x + 4, y - 8), str(index + 1),
                              fill=colors[index])

            x, y = severity * 224, row * 250
            label = f"{kind}: clean" if severity == 0 else f"{kind}: s{severity}"
            if kind == "rotation" and severity:
                label += f" ({angle:+g} deg)"
            headings.text((x + 5, y + 5), label, fill="black")
            montage.paste(overlay, (x, y + 26))

            audits.append({
                "sample_id": name,
                "kind": kind,
                "severity": severity,
                "maximum_coordinate_error_px": error,
                "gt_in_frame": in_frame.tolist(),
                "saved_image_recreated_exactly": True,
            })

    destination = output_dir / f"{Path(name).stem}_gt_overlay.png"
    montage.save(destination)
    print(f"{name}: 25 image/GT conditions checked; overlay saved")

audit = {
    "split": "validation",
    "sample_filenames": names,
    "conditions_checked": len(audits),
    "maximum_coordinate_error_px": maximum_coordinate_error,
    "rotation_directions_present": sorted(rotation_signs),
    "out_of_frame_point_occurrences": sum(
        sum(not flag for flag in row["gt_in_frame"]) for row in audits
    ),
    "checks": audits,
    "day14_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_verification": source_verification,
    "visual_review": "pending",
    "scope": (
        "Numeric transform/reconstruction checks and real-image GT overlays. "
        "Visual anatomical alignment requires manual review. "
        "GT points remain available under occlusion."
    ),
    "detector_run": False,
    "model_inference_run": False,
    "test_evaluated": False,
    "passed": True,
}
destination = output_dir / "gt_overlay_audit.json"
destination.write_text(
    json.dumps(audit, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print("Conditions checked:", len(audits))
print(f"Maximum coordinate error: {maximum_coordinate_error:.12f} px")
print("Rotation directions:", sorted(rotation_signs))
print("Out-of-frame point occurrences:", audit["out_of_frame_point_occurrences"])
print("Saved:", destination)
print("Day 15 numeric GT audit: PASS")
