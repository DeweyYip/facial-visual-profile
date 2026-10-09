"""Restore the original Day19 execution from Drive and resume verified shards."""
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import types
import zipfile


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_file(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f"Existing file differs; preserved: {path}"
    else:
        temporary = path.with_suffix(path.suffix + ".restoring")
        temporary.write_bytes(data)
        temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore_and_resume(project="/content/drive/MyDrive/FacialVisualProfile"):
    project = Path(project)
    output = project / "runs/benchmark/day19"
    binding_path = output / "execution_binding.json"
    assert binding_path.is_file(), f"Missing original binding: {binding_path}"
    original = json.loads(binding_path.read_bytes())

    # Restore the previously verified detector package only.
    version = original["environment"]["facenet-pytorch"]
    try:
        installed = importlib.metadata.version("facenet-pytorch")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps",
                               "facenet-pytorch==" + version])
        importlib.invalidate_caches()

    import numpy as np
    import torch
    assert torch.cuda.is_available(), "Select a T4 GPU runtime before restoring."
    current = {
        "python": platform.python_version(), "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "environment": {name: importlib.metadata.version(name)
                        for name in original["environment"]},
    }
    differences = []
    for name, value in current.items():
        if name == "environment":
            differences.extend((package, original[name][package], version_now)
                               for package, version_now in value.items()
                               if original[name][package] != version_now)
        elif original[name] != value:
            differences.append((name, original[name], value))
    if differences:
        print("Runtime differs from the original execution:", flush=True)
        for name, before, now in differences:
            print(f"{name}: original={before!r}; current={now!r}", flush=True)
        raise RuntimeError("Environment comparison failed before test inference. "
                           "Paste these differences back; existing results are preserved.")
    print("Original execution environment: MATCH", flush=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(original["torch_threads"])
    torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = original["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = original["tf32"]
    torch.backends.cudnn.allow_tf32 = original["tf32"]

    # Restore frozen files from the original execution's source archive.
    saved_sources = output / "source/frozen_inputs"
    handoff = json.loads((saved_sources / "day19_handoff_manifest.json").read_bytes())
    workspace = Path("/content/day19_benchmark_workspace")
    for name, expected in handoff["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        source = saved_sources / relative
        assert sha_file(source) == expected, f"Archived frozen source changed: {name}"
        restore_file(workspace / relative, source.read_bytes())
    restore_file(workspace / "day19_handoff_manifest.json",
                 (saved_sources / "day19_handoff_manifest.json").read_bytes())
    runner_path = output / "source/scripts/day19_evaluate_benchmark.py"
    assert sha_file(runner_path) == original["runner_sha256"]
    runner = load_module("day19_original_runner", runner_path)
    config_path = workspace / "configs/benchmark_v1.json"
    assert sha_file(config_path) == original["config_sha256"]
    config = json.loads(config_path.read_bytes())
    manifest_path = workspace / config["official_test_manifest"]
    assert sha_file(manifest_path) == original["manifest_sha256"]
    with manifest_path.open(newline="") as stream:
        test_names = [row["sample_id"] for row in csv.DictReader(stream)]
    assert len(test_names) == len(set(test_names)) == 19962

    models, provenance = {}, {}
    specs = {
        "r18_seed0": ("models/resnet18.py", "ResNet18Attributes"),
        "r50_seed0": ("models/resnet50.py", "ResNet50Attributes"),
        "m3_seed0": ("models/resnet18_geometry.py", "ResNet18GeometryAttributes"),
    }
    for run, (relative, class_name) in specs.items():
        print(f"Restoring {run}: verifying snapshot and checkpoint...", flush=True)
        run_root = project / "runs" / run
        training = json.loads((run_root / "config.json").read_bytes())
        assert training["attribute_names"] == config["attribute_names"]
        for name, expected in training["source_sha256"].items():
            assert sha_file(run_root / "source" / name) == expected, name
        selected = config["threshold_bindings"][run]
        checkpoint_path = run_root / selected["checkpoint"]
        assert sha_file(checkpoint_path) == selected["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == int(Path(selected["checkpoint"]).stem.split("_")[-1])
        assert checkpoint["config"]["source_sha256"] == training["source_sha256"]
        assert checkpoint["config"]["attribute_names"] == config["attribute_names"]
        model_source = run_root / "source" / relative
        module = load_module("day19_restored_" + run, model_source)
        model = getattr(module, class_name)(num_attributes=24, pretrained=False)
        keys = [key for key in ("model_state", "model_state_dict") if key in checkpoint]
        assert len(keys) == 1
        model.load_state_dict(checkpoint[keys[0]], strict=True)
        models[run] = model.to("cuda").eval()
        provenance[run] = {
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "model_source_sha256": sha_file(model_source), "state_key": keys[0],
        }
        assert provenance[run] == original["models"][run]
        del checkpoint
        print(f"{run}: strict restoration PASS", flush=True)

    package_name = "day19_restored_datasets"
    package = types.ModuleType(package_name)
    package.__path__ = [str(workspace / "datasets")]
    sys.modules[package_name] = package
    preprocessing = importlib.import_module(package_name + ".benchmark_preprocessing")
    corruptions = importlib.import_module(package_name + ".corruptions")
    detector_module = load_module("day19_restored_mtcnn", workspace / "detectors/mtcnn.py")
    detector = detector_module.MTCNNDetector(device="cpu")
    weights = hashlib.sha256()
    for name, tensor in sorted(detector.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        weights.update(name.encode("utf-8"))
        weights.update(str(value.dtype).encode("ascii"))
        weights.update(str(value.shape).encode("ascii"))
        weights.update(value.tobytes())
    assert weights.hexdigest() == original["detector_weights_sha256"]
    print("Original detector weights: MATCH", flush=True)

    data_root = Path("/content/day19_benchmark_data/raw")
    image_root = data_root / "img_align_celeba"
    image_root.mkdir(parents=True, exist_ok=True)
    directories = (project / "data", project / "data/raw", project / "data/celeba", project)
    for name, expected in original["annotation_sha256"].items():
        destination = data_root / name
        if destination.exists():
            assert sha_file(destination) == expected
            continue
        matches = [directory / name for directory in directories
                   if (directory / name).is_file() and sha_file(directory / name) == expected]
        assert matches, f"Cannot locate original annotation: {name}"
        restore_file(destination, matches[0].read_bytes())

    jpeg_manifest_path = output / "test_jpeg_sha256.json"
    assert sha_file(jpeg_manifest_path) == original["test_jpeg_manifest_sha256"]
    jpeg_hashes = json.loads(jpeg_manifest_path.read_bytes())
    assert set(jpeg_hashes) == set(test_names)
    missing_images = [name for name in test_names if not (image_root / name).is_file()]
    if missing_images:
        local_zip = Path("/content/day19_img_align_celeba.zip")
        if local_zip.exists():
            assert sha_file(local_zip) == original["image_zip_sha256"]
        else:
            candidates = [directory / "img_align_celeba.zip" for directory in directories
                          if (directory / "img_align_celeba.zip").is_file()]
            assert len(candidates) == 1, f"Ambiguous image ZIP: {candidates}"
            print("Restoring local image ZIP from Drive...", flush=True)
            temporary = local_zip.with_suffix(".copying")
            digest = hashlib.sha256()
            copied, next_progress = 0, 256 * 1024 * 1024
            with candidates[0].open("rb") as source, temporary.open("wb") as target:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    target.write(chunk)
                    copied += len(chunk)
                    if copied >= next_progress:
                        print(f"Copied {copied / 1024**2:.0f} MiB", flush=True)
                        next_progress += 256 * 1024 * 1024
            assert digest.hexdigest() == original["image_zip_sha256"]
            temporary.replace(local_zip)
        wanted = set(missing_images)
        with zipfile.ZipFile(local_zip) as archive:
            members = {}
            for member in archive.infolist():
                name = member.filename.rsplit("/", 1)[-1]
                if not member.is_dir() and name in wanted:
                    assert name not in members, name
                    members[name] = member.filename
            assert set(members) == wanted
            for index, name in enumerate(missing_images, 1):
                data = archive.read(members[name])
                assert hashlib.sha256(data).hexdigest() == jpeg_hashes[name], name
                restore_file(image_root / name, data)
                if index % 2000 == 0 or index == len(missing_images):
                    print(f"Restored test JPEGs: {index}/{len(missing_images)}", flush=True)
    for name in test_names:
        assert sha_file(image_root / name) == jpeg_hashes[name], name
    assert {path.name for path in image_root.iterdir() if path.is_file()} == set(test_names)
    print("Original 19962 test JPEG hashes: MATCH", flush=True)

    dataset_source = project / "runs/m3_seed0/source/datasets/celeba.py"
    assert sha_file(dataset_source) == original["dataset_source_sha256"]
    aliases = ("datasets", "datasets.preprocessing")
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules["datasets"] = package
        sys.modules["datasets.preprocessing"] = importlib.import_module(package_name + ".preprocessing")
        module = load_module("day19_restored_celeba", dataset_source)
    finally:
        for name, before in previous.items():
            if before is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = before
    previous_directory = Path.cwd()
    try:
        os.chdir(workspace)
        dataset = module.CelebAAttributes(split="test", root=data_root)
    finally:
        os.chdir(previous_directory)
    assert dataset.names == test_names and dataset.attribute_names == config["attribute_names"]
    print("Day19 runtime restoration: PASS; resuming original runner", flush=True)
    return runner.run_day19(
        workspace=workspace, project=project, dataset=dataset, models=models,
        detector=detector, preprocessing=preprocessing, corruptions=corruptions,
        model_provenance=provenance, detector_weights_sha256=weights.hexdigest(),
        annotation_sha256=original["annotation_sha256"],
        image_zip_sha256=original["image_zip_sha256"],
    )
