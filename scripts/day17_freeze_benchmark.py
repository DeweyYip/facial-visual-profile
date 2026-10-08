"""Freeze partition metadata and rules without accessing test images/labels."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import io
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 162770, "validation": 19867, "test": 19962}
KINDS = ["blur", "brightness", "rotation", "occlusion", "jpeg"]
LEVELS = {"blur": [1., 2., 3., 4.], "brightness": [.8, .6, .4, .2],
          "rotation": [5., 10., 15., 20.], "occlusion": [.1, .2, .3, .4],
          "jpeg": [75, 50, 25, 10]}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_frozen(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing frozen file differs: {path}; use a new protocol version")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path)
    args = parser.parse_args()
    assert (ROOT / "docs/training_protocol_v1.md").is_file()

    # Only locate/read the official partition text; never images or annotations.
    if args.partition is None:
        candidates = list((ROOT / "data").rglob("list_eval_partition.txt"))
        if len(candidates) != 1:
            raise RuntimeError("Supply --partition /absolute/path/list_eval_partition.txt; "
                               f"found {len(candidates)} candidates under data/")
        partition = candidates[0]
    else:
        partition = args.partition.expanduser().resolve()
    partition_bytes = partition.read_bytes()
    membership = {}
    for line in partition_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        assert len(fields) == 2, "Malformed official partition row"
        filename, split = fields
        assert filename not in membership, "Duplicate partition ID"
        assert split in ("0", "1", "2"), "Unknown split"
        membership[filename] = int(split)
    assert set(membership) == {f"{i:06d}.jpg" for i in range(1, 202600)}
    counts = {name: sum(value == index for value in membership.values())
              for index, name in enumerate(EXPECTED)}
    assert counts == EXPECTED, counts
    filenames = sorted(name for name, split in membership.items() if split == 2)

    draft = read_json("configs/corruptions_draft_v1.json")
    assert draft["levels"] == LEVELS
    assert draft["implementation_version"] == "day14_draft_v1"
    assert draft["corruption_seed"] == 0
    tree = ast.parse((ROOT / "datasets/corruptions.py").read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("VERSION", "LEVELS"):
                    constants[target.id] = ast.literal_eval(node.value)
    assert constants["VERSION"] == draft["implementation_version"]
    assert {key: list(value) for key, value in constants["LEVELS"].items()} == LEVELS

    quality_path = "docs/results/corruption_preparation/day16/detector_quality_report.json"
    model_path = "docs/results/corruption_preparation/day16/m3_pipeline_smoke_report.json"
    quality, model = read_json(quality_path), read_json(model_path)
    for evidence in (quality, model):
        assert evidence["passed"] is True and evidence["test_evaluated"] is False
    assert quality["runtime_errors"] == 0
    assert model["detector_report_sha256"] == sha((ROOT / quality_path).read_bytes())
    for sample in quality["sample_filenames"]:
        assert membership[sample] == 1, "Preview sample is outside official validation"
    for filename, expected in quality["source_sha256"].items():
        assert sha((ROOT / filename).read_bytes()) == expected, f"Source changed: {filename}"
    assert sha((ROOT / "evaluation/attributes.py").read_bytes()) == (
        "d45ebd6e28b94dad0af46547f8da04130f0f193289f4684b1c8540e46f1026bd")

    smoke = subprocess.run([sys.executable, str(ROOT / "scripts/day17_metrics_smoke.py")],
                           cwd=ROOT, capture_output=True, text=True)
    print(smoke.stdout, end="")
    if smoke.returncode:
        raise RuntimeError(smoke.stderr or smoke.stdout)

    # This CSV plus the condition list defines all planned sample-condition pairs.
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["sample_id", "official_split", "rotation_sign"])
    signs = []
    for filename in filenames:
        key = f"{constants['VERSION']}|0|{filename}|rotation".encode()
        sign = 1 if hashlib.sha256(key).digest()[0] & 1 else -1
        writer.writerow([filename, 2, sign])
        signs.append(sign)
    samples_bytes = buffer.getvalue().encode()
    conditions = [{"condition": "clean", "severity": 0,
                   "implementation_kind": "blur", "level": None}]
    conditions += [{"condition": kind, "severity": severity,
                    "implementation_kind": kind, "level": LEVELS[kind][severity - 1]}
                   for kind in KINDS for severity in range(1, 5)]

    attributes = []
    for line in (ROOT / "configs/attributes.yaml").read_text().splitlines():
        if line.strip().startswith("- "):
            attributes.append(line.strip()[2:].strip())
    assert len(attributes) == len(set(attributes)) == 24
    assert attributes == model["attribute_names"]

    # Freeze existing values; do not select thresholds here.
    selected = {"r18_seed0": "epoch_03.pt", "r50_seed0": "epoch_03.pt",
                "m3_seed0": "epoch_02.pt"}
    threshold_bindings = {}
    for run, checkpoint in selected.items():
        filename = f"configs/thresholds/{run}.json"
        record = read_json(filename)
        assert record["checkpoint"] == checkpoint, run
        assert record["samples"] == 19867 and record["fit_split"] == "valid", run
        assert record["attribute_names"] == attributes, run
        assert record["objective"] == "per_attribute_F1", run
        assert record["grid"] == [i / 100 for i in range(5, 96)], run
        assert record["tie_break"] == "closest_to_0.5_then_lower", run
        values = record["thresholds"]
        assert len(values) == 24 and all(value in record["grid"] for value in values)
        digest = record["checkpoint_sha256"]
        assert isinstance(digest, str) and len(digest) == 64
        assert all(char in "0123456789abcdef" for char in digest)
        threshold_bindings[run] = {
            "path": filename, "file_sha256": sha((ROOT / filename).read_bytes()),
            "checkpoint": checkpoint, "checkpoint_sha256": digest,
            "thresholds": values,
            "fit_pipeline": "historical clean validation; original-image cache for M3"}
    reproduction = read_json(
        "docs/results/benchmark_preparation/day17/r18_threshold_reproduction.json")
    for key in ("checkpoint_sha256", "attribute_names", "thresholds", "grid", "tie_break"):
        assert reproduction[key] == read_json("configs/thresholds/r18_seed0.json")[key]
    for run in ("r50_seed0", "m3_seed0"):
        archived = ROOT / f"docs/results/benchmark_preparation/day17/{run}.json"
        assert archived.read_bytes() == (ROOT / f"configs/thresholds/{run}.json").read_bytes()
        assert json.loads(archived.read_bytes())["test_evaluated"] is False
    assert read_json("configs/thresholds/m3_seed0.json")["masked_validation_samples"] == 32

    detector_constants = {}
    detector_tree = ast.parse((ROOT / "detectors/mtcnn.py").read_text())
    for node in detector_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MTCNNDetector":
            for assignment in node.body:
                if isinstance(assignment, ast.Assign):
                    for target in assignment.targets:
                        if isinstance(target, ast.Name) and target.id in ("CONFIG", "POINT_ORDER"):
                            detector_constants[target.id] = ast.literal_eval(assignment.value)
    assert set(detector_constants) == {"CONFIG", "POINT_ORDER"}
    assert list(detector_constants["POINT_ORDER"]) == quality["point_order"]

    config = dict(draft)
    config.update({
        "protocol_id": "benchmark_v1",
        "status": "sample_condition_and_metric_rules_frozen",
        "preview_split": "validation",
        "test_access": "partition_metadata_only_during_freeze",
        "official_test_manifest": "docs/results/benchmark_freeze/day17/test_samples.csv",
        "manifest_format": "sample CSV crossed with ordered conditions",
        "test_sample_count": len(filenames), "conditions": conditions,
        "planned_input_count": len(filenames) * len(conditions),
        "sample_manifest_sha256": sha(samples_bytes),
        "attribute_names": attributes,
        "detector_config": detector_constants["CONFIG"],
        "point_order": quality["point_order"],
        "tested_detector_provenance": {
            key: value for key, value in quality.items()
            if key not in ("records", "summaries")},
        "tested_detector_environment": quality["environment"],
        "tested_detector_device": quality["device"],
        "checkpoint_selection": "clean validation mAP; earliest epoch on exact tie",
        "selected_seed0_checkpoints": {
            "r18": "runs/r18_seed0/epoch_03.pt",
            "r50": "runs/r50_seed0/epoch_03.pt",
            "m3": "runs/m3_seed0/epoch_02.pt"},
        "oracle": "independent clean training and validation selection pending",
        "threshold_rule": {
            "id": "reuse_verified_historical_validation_thresholds", "split": "validation",
            "sample_count": 19867, "condition": "historical clean validation",
            "grid_integer_hundredths": list(range(5, 96)),
            "objective": "per-attribute F1",
            "tie": "closest to 0.50, then smaller; reproduced against existing R18 results",
            "comparison": ">=", "model_specific_values": "saved and hash-bound for three seed0 models",
            "reuse": "same values across all test conditions for each checkpoint"},
        "threshold_bindings": threshold_bindings,
        "m3_clean_reference": "fresh detection on final clean 224 RGB; historical thresholds fixed",
        "m3_pipeline_change": "original-image cache versus final-canvas detection; report explicitly",
        "f1_reports": ["validation_selected_thresholds", "fixed_0.5"],
        "metric_implementation": "evaluation/benchmark_attributes.py",
        "execution_errors": "record and resolve; do not count as no_face",
        "actual_input_hashes": "record during cache generation; not available at freeze",
        "test_evaluated": False,
        "note": "Rules frozen after Day 14-16 validation checks. Preserve implementation "
                "version because it determines rotation signs. Reuse three saved threshold "
                "files. Oracle training/thresholds and actual benchmark caches remain pending."
    })
    destination = ROOT / "docs/results/benchmark_freeze/day17"
    samples_path = destination / "test_samples.csv"
    config_path = ROOT / "configs/benchmark_v1.json"
    write_frozen(samples_path, samples_bytes)
    write_frozen(config_path, encoded(config))

    bound_files = list(quality["source_sha256"]) + [
        "evaluation/attributes.py", "evaluation/benchmark_attributes.py",
        "configs/attributes.yaml", "configs/benchmark_v1.json",
        "scripts/day17_freeze_benchmark.py", "scripts/day17_metrics_smoke.py",
        "docs/benchmark_protocol_v1.md", "docs/training_protocol_v1.md",
        "docs/results/benchmark_freeze/day17/test_samples.csv",
        quality_path, model_path] + [
        f"configs/thresholds/{run}.json" for run in selected] + [
        "docs/results/benchmark_preparation/day17/r18_threshold_reproduction.json",
        "docs/results/benchmark_preparation/day17/r50_seed0.json",
        "docs/results/benchmark_preparation/day17/m3_seed0.json"]
    hashes = {name: sha((ROOT / name).read_bytes()) for name in bound_files}
    audit_path = destination / "freeze_audit.json"
    previous_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    parent_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if previous_audit:
        parent_commit = previous_audit["parent_git_commit"]
    audit = {
        "protocol_id": "benchmark_v1", "parent_git_commit": parent_commit,
        "partition_sha256": sha(partition_bytes), "official_split_counts": counts,
        "test_sample_count": len(filenames), "condition_count": len(conditions),
        "planned_input_count": len(filenames) * len(conditions),
        "rotation_sign_counts": {"positive": signs.count(1), "negative": signs.count(-1)},
        "source_and_artifact_sha256": hashes,
        "day16_sources_unchanged": True, "historical_metrics_unchanged": True,
        "synthetic_metrics_smoke_stdout": smoke.stdout,
        "test_access_this_script": "official partition metadata only",
        "test_images_opened": False, "test_labels_read": False,
        "detector_run": False, "model_inference_run": False,
        "test_evaluated": False, "threshold_rule_frozen": True,
        "model_specific_threshold_values_frozen": True,
        "thresholds_frozen_for": list(selected), "oracle_thresholds_frozen": False,
        "historical_test_access": "clean original-image MTCNN test cache exists; no test attribute evaluation reported",
        "oracle_trained": False, "actual_test_inputs_generated": False,
        "passed": True}
    write_frozen(audit_path, encoded(audit))

    marker = "<!-- day17-benchmark-freeze -->"
    entry = f"""{marker}
## Day 17 — Benchmark protocol v1 frozen

- Official test manifest: 19,962 unique IDs; 21 conditions; 419,202 planned inputs.
- CSV fixes test IDs/order and deterministic rotation signs; config fixes conditions.
- Saved R18/R50/M3 thresholds are hash-bound; grid 0.05..0.95;
  closest to 0.50 then smaller on ties. Existing R18 values reproduced.
- Also report fixed-0.5 F1; preserve historical training records.
- M3 benchmark clean baseline uses fresh final-canvas detection, matching
  corruption. Historical thresholds remain fixed; pipeline change is explicit.
- M3 retains failed detections with masked geometry; no GT substitution or deletion.
- This freeze accessed partition metadata only. Earlier clean MTCNN test cache
  exists; test attribute evaluation remains pending.
- Evidence: docs/results/benchmark_freeze/day17/freeze_audit.json.
- Rules: docs/benchmark_protocol_v1.md; configs/benchmark_v1.json.
- Next: independent Oracle training/validation thresholds, final-canvas landmark
  caches, then evaluation under the frozen protocol.
"""
    for name in ("docs/experiment_log.md", "docs/current_state_v1.md"):
        path = ROOT / name
        content = path.read_text(encoding="utf-8")
        if marker not in content:
            path.write_text(content.rstrip() + "\n\n" + entry, encoding="utf-8")
    print("Official split counts:", counts)
    print("Test sample manifest:", len(filenames))
    print("Conditions:", len(conditions))
    print("Planned inputs:", len(filenames) * len(conditions))
    print("Saved R18/R50/M3 thresholds: VERIFIED and FROZEN; no refitting")
    print("Saved:", audit_path)
    print("Day 17 benchmark rule freeze: PASS")


if __name__ == "__main__":
    main()
