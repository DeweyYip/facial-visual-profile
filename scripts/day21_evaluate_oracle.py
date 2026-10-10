"""Evaluate frozen Oracle on 21 conditions using verified Day19/20 inputs."""
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
        assert path.read_bytes() == data, f"Existing file differs; preserved: {path}"
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(data)
    assert not path.exists(), f"Concurrent writer detected: {path}"
    temporary.replace(path)


def save_prediction(path, arrays, binding_sha, condition, reference_sha):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    data = buffer.getvalue()
    manifest = {
        "binding_sha256": binding_sha,
        "condition": condition,
        "reference_shard_sha256": reference_sha,
        "files": {"predictions.npz": sha(data)},
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("predictions.npz", data)
        archive.writestr("manifest.json", encoded(manifest))
    write_once(path, output.getvalue())


def load_prediction(path, binding_sha, condition, reference_sha, names, targets,
                    reference_records, preprocessing):
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == 2
        assert set(archive.namelist()) == {"predictions.npz", "manifest.json"}
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["binding_sha256"] == binding_sha
        assert manifest["condition"] == condition
        assert manifest["reference_shard_sha256"] == reference_sha
        assert set(manifest["files"]) == {"predictions.npz"}
        data = archive.read("predictions.npz")
        assert sha(data) == manifest["files"]["predictions.npz"]
    with np.load(io.BytesIO(data), allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}

    assert arrays["filenames"].tolist() == names
    np.testing.assert_array_equal(arrays["targets"], targets)
    assert arrays["oracle_logits"].shape == (len(names), 24)
    assert arrays["oracle_logits"].dtype == np.float32
    assert np.isfinite(arrays["oracle_logits"]).all()
    assert arrays["geometry_valid"].dtype == np.bool_
    assert arrays["geometry_valid"].shape == (len(names),)
    assert arrays["geometry_valid"].all()
    assert arrays["gt_pixels"].shape == (len(names), 5, 2)
    assert arrays["gt_normalized_points"].shape == (len(names), 10)
    assert np.isfinite(arrays["gt_pixels"]).all()
    assert np.isfinite(arrays["gt_normalized_points"]).all()

    assert arrays["final_rgb_sha256"].tolist() == [
        row["final_rgb_sha256"] for row in reference_records
    ]
    np.testing.assert_allclose(
        arrays["gt_pixels"],
        [row["gt_pixels"] for row in reference_records],
        rtol=0, atol=1e-12,
    )
    for index in range(len(names)):
        np.testing.assert_array_equal(
            arrays["gt_normalized_points"][index],
            preprocessing.normalize_points(arrays["gt_pixels"][index]).numpy(),
        )
    return arrays


def run_oracle(*, workspace, project, dataset, model, preprocessing, corruptions):
    workspace, project = Path(workspace), Path(project)
    output = project / "runs/benchmark/day21"
    write_once(output / "source/scripts/day21_evaluate_oracle.py",
               Path(__file__).read_bytes())

    config_path = workspace / "configs/benchmark_v1.json"
    config = json.loads(config_path.read_bytes())
    binding_path = workspace / "configs/oracle_benchmark_binding_v1.json"
    oracle_binding = json.loads(binding_path.read_bytes())
    additional = oracle_binding["additional_model"]

    for path_key, hash_key in (
        ("base_config_path", "base_config_sha256"),
        ("base_freeze_audit_path", "base_freeze_audit_sha256"),
        ("test_manifest_path", "test_manifest_sha256"),
    ):
        assert sha_file(workspace / oracle_binding[path_key]) == oracle_binding[hash_key]

    freeze = json.loads(
        (workspace / oracle_binding["base_freeze_audit_path"]).read_bytes()
    )
    assert freeze["passed"]
    for name, expected in freeze["source_and_artifact_sha256"].items():
        assert sha_file(workspace / name) == expected, name

    threshold_path = workspace / additional["threshold_path"]
    assert sha_file(threshold_path) == additional["threshold_file_sha256"]
    thresholds = json.loads(threshold_path.read_bytes())
    assert thresholds["thresholds"] == additional["thresholds"]
    assert thresholds["attribute_names"] == config["attribute_names"]
    assert thresholds["fit_split"] == "valid"
    assert thresholds["test_evaluated"] is False

    oracle_root = project / "runs/oracle_seed0"
    training_path = oracle_root / "config.json"
    training = json.loads(training_path.read_bytes())
    assert sha_file(training_path) == thresholds["config_sha256"]
    assert training["source_sha256"] == additional["source_sha256"]
    for name, expected in additional["source_sha256"].items():
        assert sha_file(oracle_root / "source" / name) == expected, name
    assert sha_file(Path(additional["checkpoint_path"])) == additional["checkpoint_sha256"]
    assert thresholds["checkpoint_sha256"] == additional["checkpoint_sha256"]

    base_reports, base_bindings, references = {}, {}, {}
    base_report_hashes, base_binding_hashes = {}, {}
    for day, count in (("day19", 9), ("day20", 12)):
        directory = project / "runs/benchmark" / day
        report_path = directory / (day + "_benchmark_report.json")
        execution_path = directory / "execution_binding.json"
        result = json.loads(report_path.read_bytes())
        execution = json.loads(execution_path.read_bytes())
        assert result[day + "_scope_complete"] is True
        assert result["condition_count"] == count
        assert result["test_sample_count"] == 19962
        assert result["runtime_errors"] == 0
        assert result["threshold_fitting_performed"] is False
        assert result["attribute_names"] == config["attribute_names"]
        assert sha_file(execution_path) == result["execution_binding_sha256"]
        base_reports[day], base_bindings[day] = result, execution
        base_report_hashes[day] = sha_file(report_path)
        base_binding_hashes[day] = sha_file(execution_path)
        for item in result["conditions"]:
            condition = item["condition"]
            label = f"{condition['condition']}_s{condition['severity']}"
            assert label not in references
            references[label] = (day, condition)

    original = base_bindings["day19"]
    assert base_bindings["day20"]["day19_report_sha256"] == base_report_hashes["day19"]
    assert base_bindings["day20"]["day19_execution_binding_sha256"] == base_binding_hashes["day19"]
    for key in original:
        if key not in ("runner_sha256", "conditions"):
            assert original[key] == base_bindings["day20"][key], key

    expected_conditions = (
        [("clean", 0)]
        + [(kind, severity) for kind in
           ("blur", "brightness", "rotation", "occlusion", "jpeg")
           for severity in range(1, 5)]
    )
    conditions = config["conditions"]
    assert [(item["condition"], item["severity"]) for item in conditions] == expected_conditions
    for condition in conditions:
        label = f"{condition['condition']}_s{condition['severity']}"
        assert references[label][1] == condition

    assert len(dataset.names) == len(set(dataset.names)) == 19962
    assert dataset.attribute_names == config["attribute_names"]
    import csv
    with (workspace / oracle_binding["test_manifest_path"]).open(newline="") as stream:
        names = [row["sample_id"] for row in csv.DictReader(stream)]
    assert names == dataset.names
    targets = np.stack([dataset.targets[name].numpy() for name in names]).astype(np.float32)

    for name, expected in original["annotation_sha256"].items():
        assert sha_file(dataset.root / name) == expected, name
    jpeg_path = project / "runs/benchmark/day19/test_jpeg_sha256.json"
    assert sha_file(jpeg_path) == original["test_jpeg_manifest_sha256"]
    jpeg_hashes = json.loads(jpeg_path.read_bytes())
    assert set(jpeg_hashes) == set(names)
    write_once(output / "test_jpeg_sha256.json", jpeg_path.read_bytes())

    assert torch.cuda.is_available() and not model.training
    assert not torch.is_autocast_enabled()
    assert torch.get_num_threads() == original["torch_threads"]
    assert torch.backends.cudnn.deterministic == original["cudnn_deterministic"]
    assert torch.backends.cudnn.benchmark == original["cudnn_benchmark"]
    assert torch.backends.cuda.matmul.allow_tf32 == original["tf32"]
    assert torch.backends.cudnn.allow_tf32 == original["tf32"]
    assert all(p.device.type == "cuda" and p.dtype == torch.float32
               for p in model.parameters())

    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    for key, value in environment.items():
        assert value == original[key], key
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "Pillow", "scikit-learn")}
    for name, version in packages.items():
        assert version == original["environment"][name], name

    execution = {
        "runner_sha256": sha_file(__file__),
        "oracle_binding_sha256": sha_file(binding_path),
        "checkpoint_sha256": additional["checkpoint_sha256"],
        "threshold_file_sha256": sha_file(threshold_path),
        "training_config_sha256": sha_file(training_path),
        "source_sha256": additional["source_sha256"],
        "config_sha256": sha_file(config_path),
        "base_report_sha256": base_report_hashes,
        "base_execution_binding_sha256": base_binding_hashes,
        "test_manifest_sha256": oracle_binding["test_manifest_sha256"],
        "test_jpeg_manifest_sha256": sha_file(jpeg_path),
        "annotation_sha256": original["annotation_sha256"],
        "image_zip_sha256": original["image_zip_sha256"],
        "conditions": conditions, "attribute_names": config["attribute_names"],
        "environment": packages, "runtime": environment,
        "precision": "FP32", "batch_size": 16, "shard_size": 256,
        "torch_threads": original["torch_threads"],
        "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False,
        "geometry_source": "transformed GT; all samples valid",
        "detector_used": False, "threshold_fitting_performed": False,
    }
    execution_bytes = encoded(execution)
    write_once(output / "execution_binding.json", execution_bytes)
    execution_sha = sha(execution_bytes)

    metric_path = workspace / "evaluation/benchmark_attributes.py"
    spec = importlib.util.spec_from_file_location("day21_frozen_metrics", metric_path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)

    summaries, shard_hashes, reference_hashes = [], {}, {}
    active = {"stage": "start"}
    started = time.perf_counter()
    print("Oracle inference starts: 21 conditions, 19962 images; no detector", flush=True)

    try:
        for condition in conditions:
            label = f"{condition['condition']}_s{condition['severity']}"
            day = references[label][0]
            logits_parts = []
            print(f"\n=== {label} ===", flush=True)

            for start in range(0, len(names), 256):
                stop = min(start + 256, len(names))
                selected = names[start:stop]
                relative = f"shards/{label}/{start:06d}_{stop:06d}.zip"
                reference_path = project / "runs/benchmark" / day / relative
                reference_sha = base_reports[day]["shard_sha256"][relative]
                active = {"condition": condition, "start": start, "stop": stop,
                          "stage": "verify_reference"}
                assert sha_file(reference_path) == reference_sha, relative
                reference_hashes[day + "/" + relative] = reference_sha

                with zipfile.ZipFile(reference_path) as archive:
                    assert len(archive.namelist()) == 3
                    assert set(archive.namelist()) == {
                        "manifest.json", "records.json", "predictions.npz"
                    }
                    manifest = json.loads(archive.read("manifest.json"))
                    assert manifest["binding_sha256"] == base_binding_hashes[day]
                    assert manifest["condition"] == condition
                    assert set(manifest["files"]) == {"records.json", "predictions.npz"}
                    records_bytes = archive.read("records.json")
                    prediction_bytes = archive.read("predictions.npz")
                    assert sha(records_bytes) == manifest["files"]["records.json"]
                    assert sha(prediction_bytes) == manifest["files"]["predictions.npz"]
                    records = json.loads(records_bytes)

                with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as stored:
                    assert stored["filenames"].tolist() == selected
                    np.testing.assert_array_equal(stored["targets"], targets[start:stop])
                assert len(records) == len(selected)
                for name, row in zip(selected, records):
                    assert row["sample_id"] == name
                    assert row["condition"] == condition["condition"]
                    assert row["severity"] == condition["severity"]

                path = output / relative
                reused = path.exists()
                if not reused:
                    images, points, gt_values, rgb_hashes = [], [], [], []
                    for name, reference in zip(selected, records):
                        active.update(stage="reconstruct_input", sample_id=name)
                        raw_bytes = (dataset.root / "img_align_celeba" / name).read_bytes()
                        assert sha(raw_bytes) == jpeg_hashes[name], name
                        with Image.open(io.BytesIO(raw_bytes)) as image:
                            canvas, gt = preprocessing.prepare_canvas(
                                image, dataset.landmarks[name]
                            )
                        image, gt, metadata = corruptions.apply_corruption(
                            canvas, condition["implementation_kind"],
                            condition["severity"], sample_id=name,
                            seed=config["corruption_seed"], gt_pixels=gt,
                        )
                        rgb_sha = corruptions.rgb_sha256(image)
                        assert rgb_sha == reference["final_rgb_sha256"], name
                        np.testing.assert_allclose(
                            gt, reference["gt_pixels"], rtol=0, atol=1e-12
                        )
                        images.append(preprocessing.normalize_canvas(image))
                        points.append(preprocessing.normalize_points(gt))
                        gt_values.append(np.asarray(gt, dtype=np.float64))
                        rgb_hashes.append(rgb_sha)

                    images, points = torch.stack(images), torch.stack(points)
                    assert torch.isfinite(images).all() and torch.isfinite(points).all()
                    logits = []
                    active.update(stage="oracle_inference")
                    with torch.inference_mode():
                        for offset in range(0, len(selected), 16):
                            end = min(offset + 16, len(selected))
                            valid = torch.ones(end - offset, dtype=torch.bool, device="cuda")
                            values = model(
                                images[offset:end].to("cuda"),
                                points[offset:end].to("cuda"), valid,
                            )
                            assert values.shape == (end - offset, 24)
                            assert torch.isfinite(values).all()
                            logits.append(values.cpu().numpy())

                    arrays = {
                        "filenames": np.asarray(selected),
                        "targets": targets[start:stop],
                        "oracle_logits": np.concatenate(logits),
                        "geometry_valid": np.ones(len(selected), dtype=np.bool_),
                        "gt_normalized_points": points.numpy(),
                        "gt_pixels": np.stack(gt_values),
                        "final_rgb_sha256": np.asarray(rgb_hashes),
                    }
                    save_prediction(path, arrays, execution_sha, condition, reference_sha)

                arrays = load_prediction(
                    path, execution_sha, condition, reference_sha,
                    selected, targets[start:stop], records, preprocessing,
                )
                shard_hashes[relative] = sha_file(path)
                logits_parts.append(arrays["oracle_logits"])
                print(
                    f"{label}: {stop}/19962; {'reused' if reused else 'saved'}; "
                    f"elapsed={(time.perf_counter()-started)/60:.1f} min",
                    flush=True,
                )

            values = np.concatenate(logits_parts)
            frozen = metrics.evaluate_benchmark_attributes(
                values, targets, additional["thresholds"]
            )
            fixed = metrics.evaluate_benchmark_attributes(values, targets, [0.5] * 24)
            summary = {
                "condition": condition, "sample_count": len(names),
                "geometry_valid_count": len(names), "detector_used": False,
                "models": {"oracle_seed0": {
                    "frozen_thresholds": frozen, "fixed_0_5": fixed,
                }},
            }
            write_once(output / "condition_reports" / (label + ".json"), encoded(summary))
            summaries.append(summary)
            print(
                f"Oracle: F1(frozen)={frozen['macro_f1']:.6f}; "
                f"F1@0.5={fixed['macro_f1']:.6f}; mAP={frozen['map']:.6f}",
                flush=True,
            )

        for summary in summaries:
            values = summary["models"]["oracle_seed0"]
            for rule in ("frozen_thresholds", "fixed_0_5"):
                clean = summaries[0]["models"]["oracle_seed0"][rule]
                values[rule + "_degradation_pp"] = {
                    metric: 100 * (clean[metric] - values[rule][metric])
                    for metric in ("macro_f1", "map")
                }

        errors = {
            path.relative_to(output).as_posix(): sha_file(path)
            for path in sorted((output / "runtime_errors").glob("*.json"))
        }
        report = {
            "scope": "Oracle on all 21 frozen conditions",
            "execution_binding_sha256": execution_sha,
            "attribute_names": config["attribute_names"],
            "test_sample_count": len(names), "condition_count": 21,
            "model_sample_evaluations": len(names) * 21,
            "test_evaluated": True, "threshold_fitting_performed": False,
            "detector_used": False, "runtime_errors": 0,
            "prior_runtime_error_attempts": errors,
            "prior_error_resolution": "All required shards verified before completion",
            "conditions": summaries, "shard_sha256": shard_hashes,
            "reference_shard_sha256": reference_hashes,
            "oracle_scope_complete": True,
            "comparison_analysis_complete": False,
            "degradation_unit": "percentage points",
        }
        assert len(shard_hashes) == len(reference_hashes) == 1638
        write_once(output / "day21_oracle_report.json", encoded(report))
        print("Day21 Oracle formal evaluation: PASS", flush=True)
        return report

    except Exception:
        error = {
            "execution_binding_sha256": execution_sha,
            "active": active, "status": "runtime_error",
            "traceback": traceback.format_exc(), "completion_blocked": True,
        }
        write_once(
            output / "runtime_errors" / (uuid.uuid4().hex + ".json"),
            encoded(error),
        )
        print("Runtime error recorded; completed shards preserved.", flush=True)
        raise
