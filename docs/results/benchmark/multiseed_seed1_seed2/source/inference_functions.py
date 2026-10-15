def load_source_shard(path, binding, condition, names, targets):
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
    for run in SOURCE_RUNS:
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


def assert_parameters_equal(actual, expected, path="parameters"):
    if isinstance(actual, dict):
        assert isinstance(expected, dict) and set(actual) == set(expected), path
        for key in actual:
            assert_parameters_equal(actual[key], expected[key], path + "." + key)
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, (list, tuple)) and len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_parameters_equal(a, b, path + f"[{index}]")
    elif isinstance(actual, float) or isinstance(expected, float):
        assert math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10), path
    else:
        assert actual == expected, path


def regenerate_inputs(condition, names, source_arrays, records):
    images, oracle_points = [], []
    maximum_gt_difference = 0.0
    for index, (name, record) in enumerate(zip(names, records)):
        raw = (DATA_ROOT / "img_align_celeba" / name).read_bytes()
        assert sha(raw) == JPEG_HASHES[name], f"Test JPEG changed: {name}"
        with Image.open(io.BytesIO(raw)) as image:
            canvas, gt = preprocessing.prepare_canvas(image, DATASET.landmarks[name])
        image, gt, metadata = corruptions.apply_corruption(canvas, condition["implementation_kind"], condition["severity"], sample_id=name, seed=CONFIG["corruption_seed"], gt_pixels=gt)
        assert corruptions.rgb_sha256(image) == record["final_rgb_sha256"], f"Final RGB differs: {name}"
        assert_parameters_equal(metadata["parameters"], record["corruption_parameters"])
        gt = np.asarray(gt, dtype=np.float64)
        delta = float(np.max(np.abs(gt - np.asarray(record["gt_pixels"], dtype=np.float64))))
        assert delta <= 1e-9, f"Transformed GT differs: {name}"
        maximum_gt_difference = max(maximum_gt_difference, delta)
        valid = bool(source_arrays["geometry_valid"][index])
        if valid:
            points = preprocessing.normalize_points(record["detection"]["landmarks"]).numpy()
            np.testing.assert_allclose(points, source_arrays["predicted_normalized_points"][index], rtol=0, atol=1e-6)
            eye_distance = float(np.linalg.norm(gt[0] - gt[1]))
            assert eye_distance > 0
            nme = float(np.linalg.norm(np.asarray(record["detection"]["landmarks"], dtype=np.float64) - gt, axis=1).mean() / eye_distance)
            assert abs(nme - record["nme"]) <= 1e-9
        else:
            assert not source_arrays["predicted_normalized_points"][index].any()
            assert record["nme"] is None
        images.append(preprocessing.normalize_canvas(image))
        oracle_points.append(preprocessing.normalize_points(gt))
    images = torch.stack(images)
    oracle_points = torch.stack(oracle_points)
    assert torch.isfinite(images).all() and torch.isfinite(oracle_points).all()
    return images, oracle_points, maximum_gt_difference


def save_predictions(path, arrays, records, binding_sha, source_sha, condition, gt_difference):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = {"predictions.npz": buffer.getvalue(), "records.json": encoded(records)}
    manifest = {"binding_sha256": binding_sha, "source_shard_sha256": source_sha, "condition": condition, "maximum_gt_difference_px": gt_difference, "files": {name: sha(data) for name, data in payload.items()}}
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", encoded(manifest))
    write_once(path, result.getvalue())


def load_predictions(path, names, targets, binding_sha, source_sha, condition):
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == 3 and set(archive.namelist()) == {"predictions.npz", "records.json", "manifest.json"}
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["binding_sha256"] == binding_sha and manifest["source_shard_sha256"] == source_sha
        assert manifest["condition"] == condition
        assert set(manifest["files"]) == {"predictions.npz", "records.json"}
        payload = {name: archive.read(name) for name in manifest["files"]}
        for name, data in payload.items():
            assert sha(data) == manifest["files"][name]
    with np.load(io.BytesIO(payload["predictions.npz"]), allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    assert arrays["filenames"].tolist() == names
    np.testing.assert_array_equal(arrays["targets"], targets)
    records = json.loads(payload["records.json"])
    assert len(records) == len(names)
    assert arrays["geometry_valid"].shape == (len(names),) and arrays["geometry_valid"].dtype == np.bool_
    for key in ("predicted_normalized_points", "gt_normalized_points"):
        assert arrays[key].shape == (len(names), 10) and np.isfinite(arrays[key]).all()
    for index, record in enumerate(records):
        assert record["sample_id"] == names[index]
        assert record["condition"] == condition["condition"] and record["severity"] == condition["severity"]
        valid = record["detection"]["status"] == "ok"
        assert bool(arrays["geometry_valid"][index]) == valid
        if not valid:
            assert not arrays["predicted_normalized_points"][index].any()
    for run in RUNS:
        values = arrays[run + "_logits"]
        assert values.shape == (len(names), 24) and values.dtype == np.float32 and np.isfinite(values).all()
    assert manifest["maximum_gt_difference_px"] <= 1e-9
    return arrays, records, manifest


def run_benchmark(binding_sha):
    assert not torch.is_autocast_enabled(), "FP32 inference requires autocast disabled"
    started = time.perf_counter()
    summaries, shard_hashes = [], {}
    active = {}
    processed = 0
    try:
        for condition in CONDITIONS:
            label = f"{condition['condition']}_s{condition['severity']}"
            print("\n===", label, "===", flush=True)
            parts = {run: [] for run in RUNS}
            status_counts = Counter()
            nme_values = []
            maximum_gt_difference = 0.0
            for start in range(0, len(TEST_NAMES), SHARD_SIZE):
                stop = min(start + SHARD_SIZE, len(TEST_NAMES))
                names = TEST_NAMES[start:stop]
                source = SOURCE_INDEX[(label, start)]
                path = OUTPUT_ROOT / "shards" / label / f"{start:06d}_{stop:06d}.zip"
                active = {"condition": label, "start": start, "stop": stop}
                if not path.exists():
                    assert sha_file(source["path"]) == source["sha256"], "Seed0 source shard changed"
                    old, records = load_source_shard(source["path"], source["binding_sha256"], condition, names, TARGETS[start:stop])
                    images, gt_points, gt_difference = regenerate_inputs(condition, names, old, records)
                    predicted = torch.from_numpy(old["predicted_normalized_points"])
                    valid = torch.from_numpy(old["geometry_valid"])
                    arrays = {"filenames": np.asarray(names), "targets": TARGETS[start:stop],
                              "geometry_valid": old["geometry_valid"], "predicted_normalized_points": old["predicted_normalized_points"],
                              "gt_normalized_points": gt_points.numpy()}
                    for run in RUNS:
                        kind = run.split("_seed")[0]
                        batches = []
                        with torch.inference_mode():
                            for offset in range(0, len(names), BATCH_SIZE):
                                end = min(offset + BATCH_SIZE, len(names))
                                image_batch = images[offset:end].to("cuda")
                                if kind in ("r18", "r50"):
                                    logits = MODELS[run](image_batch)
                                else:
                                    point_batch = (predicted if kind == "m3" else gt_points)[offset:end].to("cuda")
                                    mask = valid[offset:end].to("cuda") if kind == "m3" else torch.ones(end-offset, dtype=torch.bool, device="cuda")
                                    logits = MODELS[run](image_batch, point_batch, mask)
                                    if kind == "m3" and (~mask).any():
                                        for replacement in (1e6, float("nan")):
                                            altered = point_batch.clone()
                                            altered[~mask] = replacement
                                            check = MODELS[run](image_batch, altered, mask)
                                            assert torch.equal(check, logits), f"Invalid-coordinate invariance failed: {run}"
                                assert logits.shape == (end-offset, 24) and torch.isfinite(logits).all()
                                batches.append(logits.cpu().numpy())
                        arrays[run + "_logits"] = np.concatenate(batches)
                    save_predictions(path, arrays, records, binding_sha, source["sha256"], condition, gt_difference)
                    del old, images, gt_points, predicted, valid, arrays, records, batches
                arrays, records, manifest = load_predictions(path, names, TARGETS[start:stop], binding_sha, source["sha256"], condition)
                shard_hashes[path.relative_to(OUTPUT_ROOT).as_posix()] = sha_file(path)
                maximum_gt_difference = max(maximum_gt_difference, manifest["maximum_gt_difference_px"])
                for record in records:
                    status_counts[record["detection"]["status"]] += 1
                    if record["nme"] is not None:
                        nme_values.append(record["nme"])
                for run in RUNS:
                    parts[run].append(arrays[run + "_logits"])
                processed += len(names)
                if stop % 2048 == 0 or stop == len(TEST_NAMES):
                    elapsed = (time.perf_counter() - started) / 60
                    print(f"{label}: {stop}/{len(TEST_NAMES)} verified/saved; elapsed={elapsed:.1f} min", flush=True)
                del arrays, records
            summary = {"condition": condition, "sample_count": len(TEST_NAMES), "detector_status_counts": dict(status_counts),
                       "detector_nme": {"valid_samples": len(nme_values), "mean": float(np.mean(nme_values)) if nme_values else None, "unit": "fraction of GT inter-eye distance"},
                       "maximum_gt_difference_px": maximum_gt_difference, "models": {}}
            assert sum(status_counts.values()) == 19962
            source_day = "day19" if condition["condition"] in ("clean", "blur", "brightness") else "day20"
            reference_summary = next(item for item in SOURCE_INFO[source_day]["report"]["conditions"] if item["condition"] == condition)
            for status in ("ok", "no_face", "invalid_output"):
                assert status_counts[status] == reference_summary["detector"][status]
            if nme_values:
                assert abs(float(np.mean(nme_values)) - reference_summary["detector"]["mean_nme"]) <= 1e-10
            for run in RUNS:
                logits = np.concatenate(parts[run])
                frozen = evaluate(logits, TARGETS, THRESHOLDS[run]["thresholds"])
                fixed = evaluate(logits, TARGETS, [0.5] * 24)
                assert abs(frozen["map"] - fixed["map"]) <= 1e-12
                summary["models"][run] = {"frozen_thresholds": frozen, "fixed_0_5": fixed}
                print(f"{run}: F1(frozen)={frozen['macro_f1']:.6f}; F1@0.5={fixed['macro_f1']:.6f}; mAP={frozen['map']:.6f}", flush=True)
            summaries.append(summary)
            write_once(OUTPUT_ROOT / "condition_reports" / f"{label}.json", encoded(summary))
            del parts, logits
        assert len(summaries) == 21 and len(shard_hashes) == 1638
        for summary in summaries:
            for run in RUNS:
                for rule in ("frozen_thresholds", "fixed_0_5"):
                    clean = summaries[0]["models"][run][rule]
                    current = summary["models"][run][rule]
                    summary["models"][run][rule + "_degradation_pp"] = {name: 100 * (clean[name] - current[name]) for name in ("macro_f1", "map")}
        report = {"seeds": list(SEEDS), "runs": list(RUNS), "attribute_names": CONFIG["attribute_names"],
                  "execution_binding_sha256": binding_sha, "test_sample_count": 19962, "condition_count": 21,
                  "unique_inputs": 419202, "model_sample_evaluations": 3353616,
                  "conditions": summaries, "shard_sha256": shard_hashes,
                  "detector_outputs_reused_from_seed0": True, "threshold_fitting_performed": False,
                  "test_evaluated": True, "full_benchmark_complete": True}
        write_once(OUTPUT_ROOT / "multiseed_benchmark_report.json", encoded(report))
        print("Seed1 and Seed2 four-model 21-condition test benchmark: COMPLETE")
        return report
    except Exception:
        error = {"binding_sha256": binding_sha, "active": active, "traceback": traceback.format_exc(), "completed_batches_preserved": True}
        write_once(OUTPUT_ROOT / "runtime_errors" / (uuid.uuid4().hex + ".json"), encoded(error))
        print("Execution stopped; verified completed shards remain on Drive", flush=True)
        raise
