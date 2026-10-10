"""Day20: rotation/occlusion/JPEG test evaluation; reuse verified Day19 implementation and clean reference."""
from collections import Counter
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import time
import traceback
import uuid
import zipfile

import numpy as np
from PIL import Image
import torch


RUNS = ("r18_seed0", "r50_seed0", "m3_seed0")
SHARD_SIZE = 256
BATCH_SIZE = 16


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
            + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing file differs; preserved: {path}")
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(data)
    if path.exists():
        temporary.unlink()
        raise RuntimeError(f"Concurrent writer detected: {path}")
    temporary.replace(path)


def save_shard(path, binding, condition, arrays, records):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = {"predictions.npz": buffer.getvalue(), "records.json": encoded(records)}
    manifest = {"binding_sha256": binding, "condition": condition,
                "files": {name: sha(data) for name, data in payload.items()}}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", encoded(manifest))
    write_once(path, output.getvalue())


def load_shard(path, binding, condition, names, targets):
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == 3
        assert set(archive.namelist()) == {"manifest.json", "records.json", "predictions.npz"}
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["binding_sha256"] == binding
        assert manifest["condition"] == condition
        assert set(manifest["files"]) == {"records.json", "predictions.npz"}
        payload = {key: archive.read(key) for key in manifest["files"]}
        for key, data in payload.items():
            assert sha(data) == manifest["files"][key], f"Shard checksum failed: {path}"
    with np.load(io.BytesIO(payload["predictions.npz"]), allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    records = json.loads(payload["records.json"])
    assert arrays["filenames"].tolist() == names
    np.testing.assert_array_equal(arrays["targets"], targets)
    assert len(records) == len(names)
    assert arrays["geometry_valid"].shape == (len(names),)
    assert arrays["geometry_valid"].dtype == np.bool_
    assert arrays["predicted_normalized_points"].shape == (len(names), 10)
    assert np.isfinite(arrays["predicted_normalized_points"]).all()
    for run in RUNS:
        logits = arrays[run + "_logits"]
        assert logits.shape == (len(names), 24) and logits.dtype == np.float32
        assert np.isfinite(logits).all()
    for index, (name, record) in enumerate(zip(names, records)):
        assert record["sample_id"] == name
        assert record["condition"] == condition["condition"]
        assert record["severity"] == condition["severity"]
        assert record["detection"]["status"] in ("ok", "no_face", "invalid_output")
        valid = record["detection"]["status"] == "ok"
        assert bool(arrays["geometry_valid"][index]) == valid
        if valid:
            assert record["nme"] is not None and np.isfinite(record["nme"])
        else:
            assert record["nme"] is None
            assert not arrays["predicted_normalized_points"][index].any()
    return arrays, records


def run_day20(*, workspace, project, dataset, models, detector,
              preprocessing, corruptions, model_provenance,
              detector_weights_sha256, annotation_sha256, image_zip_sha256):
    workspace, project = Path(workspace), Path(project)
    output = project / "runs/benchmark/day20"
    output.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "configs/benchmark_v1.json"
    config = json.loads(config_path.read_bytes())
    freeze_path = workspace / "docs/results/benchmark_freeze/day17/freeze_audit.json"
    freeze = json.loads(freeze_path.read_bytes())
    assert freeze["passed"]
    for name, expected in freeze["source_and_artifact_sha256"].items():
        assert sha_file(workspace / name) == expected, f"Frozen file changed: {name}"
    manifest_path = workspace / config["official_test_manifest"]
    assert sha_file(manifest_path) == config["sample_manifest_sha256"]
    with manifest_path.open(newline="") as stream:
        names = [row["sample_id"] for row in csv.DictReader(stream)]
    assert names == dataset.names and len(names) == len(set(names)) == 19962
    assert dataset.attribute_names == config["attribute_names"]
    conditions = [item for item in config["conditions"]
                  if item["condition"] in ("rotation", "occlusion", "jpeg")]
    assert [(item["condition"], item["severity"]) for item in conditions] == (
        [(kind, severity) for kind in ("rotation", "occlusion", "jpeg")
                          for severity in range(1, 5)])
    quality = json.loads((workspace / "docs/results/corruption_preparation/day16/"
                         "detector_quality_report.json").read_bytes())
    weights = hashlib.sha256()
    for name, tensor in sorted(detector.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        weights.update(name.encode("utf-8"))
        weights.update(str(value.dtype).encode("ascii"))
        weights.update(str(value.shape).encode("ascii"))
        weights.update(value.tobytes())
    assert weights.hexdigest() == detector_weights_sha256 == quality["detector_weights_sha256"]
    assert detector.CONFIG == quality["detector_config"]
    assert list(detector.POINT_ORDER) == quality["point_order"]
    assert str(detector.device) == "cpu" and torch.get_num_threads() == 2
    assert set(models) == set(RUNS) and torch.cuda.is_available()
    assert not torch.is_autocast_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    for run in RUNS:
        assert not models[run].training
        assert all(parameter.device.type == "cuda" and parameter.dtype == torch.float32
                   for parameter in models[run].parameters())
        assert model_provenance[run]["checkpoint_sha256"] == (
            config["threshold_bindings"][run]["checkpoint_sha256"])
    for name, expected in annotation_sha256.items():
        assert sha_file(dataset.root / name) == expected
    assert annotation_sha256["list_eval_partition.txt"] == freeze["partition_sha256"]

    # Record the source bytes of each selected JPEG before first inference.
    # Their extraction from the recorded ZIP was verified by the preparation cell.
    print("Binding test JPEG bytes...", flush=True)
    jpeg_hashes = {name: sha_file(dataset.root / "img_align_celeba" / name) for name in names}
    jpeg_bytes = encoded(jpeg_hashes)
    write_once(output / "test_jpeg_sha256.json", jpeg_bytes)
    binding = {
        "runner_sha256": sha_file(__file__), "config_sha256": sha_file(config_path),
        "freeze_audit_sha256": sha_file(freeze_path),
        "manifest_sha256": sha_file(manifest_path), "models": model_provenance,
        "detector_weights_sha256": detector_weights_sha256,
        "annotation_sha256": annotation_sha256, "image_zip_sha256": image_zip_sha256,
        "test_jpeg_manifest_sha256": sha(jpeg_bytes),
        "dataset_source_sha256": sha_file(project / "runs/m3_seed0/source/datasets/celeba.py"),
        "conditions": conditions, "shard_size": SHARD_SIZE, "batch_size": BATCH_SIZE,
        "precision": "FP32", "detector_device": "cpu", "torch_threads": 2,
        "model_device": "cuda", "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(), "platform": platform.platform(),
        "environment": {name: importlib.metadata.version(name) for name in
                        ("torch", "torchvision", "facenet-pytorch", "numpy", "Pillow", "scikit-learn")},
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False,
        "attribute_names": config["attribute_names"],
        "failure_policy": "retain sample; zero coordinates and mask geometry after MLP",
        "threshold_fitting_performed": False,
    }
    day19_output = project / "runs/benchmark/day19"
    original_binding_path = day19_output / "execution_binding.json"
    original_binding = json.loads(original_binding_path.read_bytes())
    report_path = day19_output / "day19_benchmark_report.json"
    assert sha_file(report_path) == (
        "79e36c3725d10c2319206023f2b859854"
        "108fc7f1fc176e6998f0d96331b3e1b"
    )
    previous_report = json.loads(report_path.read_bytes())
    assert previous_report["execution_binding_sha256"] == sha_file(
        original_binding_path
    )
    assert previous_report["day19_scope_complete"] is True
    assert previous_report["threshold_fitting_performed"] is False
    assert previous_report["runtime_errors"] == 0

    assert set(binding) == set(original_binding)
    for key in binding:
        if key not in ("runner_sha256", "conditions"):
            assert binding[key] == original_binding[key], (
                f"Shared Day19/Day20 execution setting differs: {key}"
            )

    clean_candidates = [
        item for item in previous_report["conditions"]
        if item["condition"]["condition"] == "clean"
        and item["condition"]["severity"] == 0
    ]
    assert len(clean_candidates) == 1
    clean_reference = clean_candidates[0]

    binding["day19_execution_binding_sha256"] = sha_file(
        original_binding_path
    )
    binding["day19_report_sha256"] = sha_file(report_path)
    binding_bytes = encoded(binding)
    write_once(output / "execution_binding.json", binding_bytes)
    binding_sha = sha(binding_bytes)
    metric_path = workspace / "evaluation/benchmark_attributes.py"
    spec = importlib.util.spec_from_file_location("day19_frozen_metrics", metric_path)
    metric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    evaluate = metric_module.evaluate_benchmark_attributes
    targets = np.stack([dataset.targets[name].numpy() for name in names]).astype(np.float32)
    assert targets.shape == (19962, 24) and np.isin(targets, [0, 1]).all()
    assert np.all((targets.sum(axis=0) > 0) & (targets.sum(axis=0) < len(names)))
    summaries, shard_hashes = [], {}
    started, fresh_count, completed_count = time.perf_counter(), 0, 0
    active = {"stage": "start"}
    print("Official test inference starts: 12 conditions, 19962 images, 3 models", flush=True)
    try:
        for condition in conditions:
            label = f"{condition['condition']}_s{condition['severity']}"
            condition_arrays, condition_records = [], []
            print(f"\n=== {label} ===", flush=True)
            for start in range(0, len(names), SHARD_SIZE):
                stop = min(start + SHARD_SIZE, len(names))
                selected = names[start:stop]
                path = output / "shards" / label / f"{start:06d}_{stop:06d}.zip"
                active = {"condition": condition, "start": start, "stop": stop,
                          "stage": "load_or_compute_shard"}
                reused = path.exists()
                if not reused:
                    images, points, valid_flags, records = [], [], [], []
                    for name in selected:
                        active.update(sample_id=name, stage="preprocessing_and_detection")
                        raw_bytes = (dataset.root / "img_align_celeba" / name).read_bytes()
                        assert sha(raw_bytes) == jpeg_hashes[name], f"JPEG changed: {name}"
                        with Image.open(io.BytesIO(raw_bytes)) as image:
                            canvas, gt = preprocessing.prepare_canvas(image, dataset.landmarks[name])
                        image, gt, metadata = corruptions.apply_corruption(
                            canvas, condition["implementation_kind"], condition["severity"],
                            sample_id=name, seed=config["corruption_seed"], gt_pixels=gt)
                        detection = detector.detect(image)
                        status = detection["status"]
                        assert status in ("ok", "no_face", "invalid_output"), status
                        valid = status == "ok"
                        point_tensor = (preprocessing.normalize_points(detection["landmarks"])
                                        if valid else torch.zeros(10, dtype=torch.float32))
                        eye_distance = float(np.linalg.norm(gt[0] - gt[1]))
                        assert eye_distance > 0 and np.isfinite(gt).all()
                        nme = (float(np.linalg.norm(np.asarray(detection["landmarks"], dtype=np.float64)
                                                   - gt, axis=1).mean() / eye_distance)
                               if valid else None)
                        image_tensor = preprocessing.normalize_canvas(image)
                        assert torch.isfinite(image_tensor).all() and torch.isfinite(point_tensor).all()
                        records.append({"sample_id": name, "condition": condition["condition"],
                                        "severity": condition["severity"],
                                        "final_rgb_sha256": corruptions.rgb_sha256(image),
                                        "corruption_parameters": metadata["parameters"],
                                        "gt_pixels": np.asarray(gt).tolist(),
                                        "gt_eye_distance_px": eye_distance,
                                        "detection": detection, "nme": nme})
                        images.append(image_tensor)
                        points.append(point_tensor)
                        valid_flags.append(valid)
                    images = torch.stack(images)
                    points = torch.stack(points)
                    valid_flags = torch.tensor(valid_flags, dtype=torch.bool)
                    arrays = {"filenames": np.asarray(selected), "targets": targets[start:stop],
                              "geometry_valid": valid_flags.numpy(),
                              "predicted_normalized_points": points.numpy()}
                    with torch.inference_mode():
                        for run in RUNS:
                            active.update(stage="model_inference", model=run)
                            logits = []
                            for offset in range(0, len(selected), BATCH_SIZE):
                                end = min(offset + BATCH_SIZE, len(selected))
                                batch = images[offset:end].to("cuda")
                                value = (models[run](batch, points[offset:end].to("cuda"),
                                                     valid_flags[offset:end].to("cuda"))
                                         if run == "m3_seed0" else models[run](batch))
                                assert value.shape == (end - offset, 24)
                                assert torch.isfinite(value).all()
                                logits.append(value.cpu().numpy())
                            arrays[run + "_logits"] = np.concatenate(logits)
                    active.update(stage="save_shard")
                    save_shard(path, binding_sha, condition, arrays, records)
                    fresh_count += len(selected)
                arrays, records = load_shard(path, binding_sha, condition, selected, targets[start:stop])
                shard_hashes[str(path.relative_to(output))] = sha_file(path)
                condition_arrays.append(arrays)
                condition_records.extend(records)
                completed_count += len(selected)
                elapsed = time.perf_counter() - started
                eta = ((len(names) * len(conditions) - completed_count) * elapsed / fresh_count / 60
                       if fresh_count else None)
                print(f"{label}: {stop}/{len(names)}; {'reused' if reused else 'saved'}; "
                      f"elapsed={elapsed/60:.1f} min" +
                      (f"; remaining estimate={eta:.1f} min" if eta is not None else ""), flush=True)
            valid_nme = [row["nme"] for row in condition_records if row["nme"] is not None]
            statuses = Counter(row["detection"]["status"] for row in condition_records)
            assert len(condition_records) == len(names)
            summary = {"condition": condition, "sample_count": len(names),
                       "detector": {"ok": statuses["ok"], "no_face": statuses["no_face"],
                                    "invalid_output": statuses["invalid_output"], "runtime_errors": 0,
                                    "coverage": statuses["ok"] / len(names),
                                    "nme_count": len(valid_nme),
                                    "mean_nme": float(np.mean(valid_nme)) if valid_nme else None,
                                    "median_nme": float(np.median(valid_nme)) if valid_nme else None},
                       "models": {}}
            for run in RUNS:
                values = np.concatenate([item[run + "_logits"] for item in condition_arrays])
                frozen = evaluate(values, targets, config["threshold_bindings"][run]["thresholds"])
                fixed = evaluate(values, targets, [0.5] * 24)
                summary["models"][run] = {"frozen_thresholds": frozen, "fixed_0_5": fixed}
                print(f"{run}: F1(frozen)={frozen['macro_f1']:.6f}; "
                      f"F1@0.5={fixed['macro_f1']:.6f}; mAP={frozen['map']:.6f}", flush=True)
            write_once(output / "condition_reports" / (label + ".json"), encoded(summary))
            summaries.append(summary)
        for summary in summaries:
            for run in RUNS:
                for rule in ("frozen_thresholds", "fixed_0_5"):
                    clean = clean_reference["models"][run][rule]
                    current = summary["models"][run][rule]
                    summary["models"][run][rule + "_degradation_pp"] = {
                        metric: 100 * (clean[metric] - current[metric])
                        for metric in ("macro_f1", "map")}
        previous_errors = {str(path.relative_to(output)): sha_file(path)
                           for path in sorted((output / "runtime_errors").glob("*.json"))}
        report = {"scope": "Day20 rotation, occlusion, jpeg; Oracle pending",
                  "execution_binding_sha256": binding_sha, "attribute_names": config["attribute_names"],
                  "test_sample_count": len(names), "condition_count": len(conditions),
                  "unique_inputs": len(names) * len(conditions), "model_sample_evaluations": len(names) * len(conditions) * 3,
                  "test_evaluated": True, "threshold_fitting_performed": False,
                  "runtime_errors": 0, "prior_runtime_error_attempts": previous_errors,
                  "prior_error_resolution": "All required shards verified before completion",
                  "nme_unit": "ratio", "degradation_unit": "percentage points",
                  "conditions": summaries, "shard_sha256": shard_hashes,
                  "day20_scope_complete": True, "r18_r50_m3_condition_count_completed": 21, "full_benchmark_complete": False}
        write_once(output / "day20_benchmark_report.json", encoded(report))
        print(f"\nSaved: {output / 'day20_benchmark_report.json'}", flush=True)
        print("Day20 formal test evaluation: PASS", flush=True)
        return report
    except Exception:
        error = {"execution_binding_sha256": binding_sha, "active": active,
                 "status": "runtime_error", "traceback": traceback.format_exc(),
                 "completion_blocked": True}
        write_once(output / "runtime_errors" / (uuid.uuid4().hex + ".json"), encoded(error))
        print("Runtime error recorded; completed shards preserved. Completion blocked.", flush=True)
        raise
