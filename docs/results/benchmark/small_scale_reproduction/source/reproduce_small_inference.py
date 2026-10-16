"""Small fresh-session replay of Seed1 attribute models; cached detector inputs."""
from pathlib import Path
import csv
import gc
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import platform
import sys
import time
import types
import zipfile

import numpy as np
import PIL
from PIL import Image
import torch
from torch.torch_version import TorchVersion

ARCHIVE_SHA = '4c0cece5d688bbe87bb490c7f4c2429059a6ab40dae9c84b5cfd19ecb4baf45c'
LABELS = ('clean_s0', 'blur_s4', 'rotation_s4', 'occlusion_s4')
MODELS = ('r18', 'r50', 'm3', 'oracle')


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        check(path.read_bytes() == data, f'Existing file differs; preserved: {path}')
        return
    temp = path.with_name(path.name+'.partial')
    temp.write_bytes(data)
    temp.replace(path)


def sigmoid(values):
    x = np.asarray(values, dtype=np.float64)
    p = np.empty_like(x)
    positive = x >= 0
    p[positive] = 1/(1+np.exp(-x[positive]))
    e = np.exp(x[~positive]); p[~positive] = e/(1+e)
    return p


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare(reference, actual, thresholds):
    check(actual.shape == reference.shape and np.isfinite(actual).all(), 'Invalid replay logits')
    rp, ap = sigmoid(reference), sigmoid(actual)
    error = np.abs(actual.astype(np.float64)-reference.astype(np.float64))
    prob_error = np.abs(ap-rp)
    changes = {}
    for rule, threshold in (('frozen', thresholds), ('at_0_5', [.5]*24)):
        before, after = rp >= threshold, ap >= threshold
        indices = np.argwhere(before != after)
        changes[rule] = dict(changed_decisions=len(indices),
                            changed_images=int(np.any(before != after,axis=1).sum()),
                            changed_indices=indices.tolist())
    logit_close = bool(np.allclose(actual, reference, atol=1e-4, rtol=1e-4))
    probability_close = bool(np.allclose(ap, rp, atol=1e-5, rtol=1e-5))
    return dict(max_absolute_logit_error=float(error.max()), mean_absolute_logit_error=float(error.mean()),
                max_absolute_score_error=float(prob_error.max()),
                logits_within_tolerance=logit_close, scores_within_tolerance=probability_close,
                decisions=changes,
                replay_passed=logit_close and probability_close and all(v['changed_decisions']==0 for v in changes.values()))


def run(project, output, source_code):
    started = time.perf_counter()
    project, output = Path(project), Path(output)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check(PIL.__version__ == '11.3.0', 'Frozen RGB reproduction requires Pillow 11.3.0')
    base = project/'runs/benchmark/multiseed_seed1_seed2'
    raw_archive = (base/'seed1_seed2_test_results_archive.zip').read_bytes()
    check(sha(raw_archive)==ARCHIVE_SHA, 'Formal test archive SHA256 differs')
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as z:
        manifest=json.loads(z.read('archive_manifest.json'))
        for name,digest in manifest['files'].items():
            check(sha(z.read(name))==digest, f'Formal archive member differs: {name}')
        binding_bytes=z.read('execution_binding.json')
        binding=json.loads(binding_bytes)
        report=json.loads(z.read('multiseed_benchmark_report.json'))
        jpeg_hashes=json.loads(z.read('test_jpeg_sha256.json'))
        frozen=z.read('source/day19_frozen_inputs.zip')
    check(report['full_benchmark_complete'] is True, 'Reference test incomplete')
    check(sha(binding_bytes)==report['execution_binding_sha256'], 'Reference report binding differs')
    check(sha(frozen)==binding['frozen_zip_sha256'], 'Frozen preprocessing ZIP differs')
    conditions={f"{c['condition']}_s{c['severity']}":c for c in binding['conditions']}
    attributes=report['attribute_names']
    check(len(attributes)==24, 'Attribute count differs')
    # Verify original annotations against the frozen benchmark binding.
    old_binding_path=project/'runs/benchmark/day19/execution_binding.json'
    old_bytes=old_binding_path.read_bytes()
    check(sha(old_bytes)==binding['source_reports']['day19']['execution_binding.json'], 'Seed0 binding differs')
    old_binding=json.loads(old_bytes)
    annotation={}
    for name,digest in old_binding['annotation_sha256'].items():
        data=(project/'data'/name).read_bytes()
        check(sha(data)==digest, f'Annotation differs: {name}')
        annotation[name]=data.decode('utf-8-sig').splitlines()
    names=[line.split()[0] for line in annotation['list_eval_partition.txt'] if line.split()[1]=='2']
    check(len(names)==len(set(names))==19962, 'Official test membership differs')
    selected_names=names[:8]
    attr_lines=annotation['list_attr_celeba.txt']; header=attr_lines[1].split()
    columns=[header.index(a) for a in attributes]
    targets={}
    for line in attr_lines[2:]:
        parts=line.split()
        if parts[0] in selected_names:
            targets[parts[0]]=[int(parts[i+1])==1 for i in columns]
    selected_targets=np.asarray([targets[n] for n in selected_names],dtype=np.float32)
    landmarks={}
    for line in annotation['list_landmarks_align_celeba.txt'][2:]:
        parts=line.split()
        if parts[0] in selected_names:landmarks[parts[0]]=[float(v) for v in parts[1:]]
    workspace=Path('/content/small_reproduction_source')
    package_name='_fvp_repro_frozen'
    package=types.ModuleType(package_name); package.__path__=[str(workspace)]
    sys.modules[package_name]=package
    frozen_modules={}
    with zipfile.ZipFile(io.BytesIO(frozen)) as z:
        handoff=json.loads(z.read('day19_handoff_manifest.json'))
        for name in ('preprocessing','benchmark_preprocessing','corruptions'):
            relative=f'datasets/{name}.py'; data=z.read(relative)
            check(sha(data)==handoff['files'][relative], f'Frozen source differs: {relative}')
            path=workspace/(name+'.py');atomic(path,data)
            frozen_modules[name]=load_module(package_name+'.'+name,path)
    prep=frozen_modules['benchmark_preprocessing']; corruptions=frozen_modules['corruptions']
    # Prepare the fixed small image set directly from original JPEG members.
    jpeg_path=project/'data/img_align_celeba.zip'
    raw_images={}
    with zipfile.ZipFile(jpeg_path) as z:
        members={}
        for info in z.infolist():
            if not info.is_dir() and info.filename.lower().endswith('.jpg'):
                name=Path(info.filename).name
                check(name not in members,'Duplicate JPEG member')
                members[name]=info.filename
        for name in selected_names:
            data=z.read(members[name]);check(sha(data)==jpeg_hashes[name], f'JPEG differs: {name}')
            raw_images[name]=data
    inputs={}; source_shards={}
    for label in LABELS:
        condition=conditions[label]
        relative=f'shards/{label}/000000_000256.zip'
        data=(base/relative).read_bytes()
        check(sha(data)==report['shard_sha256'][relative], f'Reference shard differs: {label}')
        source_shards[relative]=sha(data)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            sm=json.loads(z.read('manifest.json'))
            check(sm['binding_sha256']==report['execution_binding_sha256'] and sm['condition']==condition,'Shard binding differs')
            check(set(z.namelist())=={'manifest.json','records.json','predictions.npz'} and len(z.namelist())==3,'Unexpected shard schema')
            payload={name:z.read(name) for name in ('records.json','predictions.npz')}
            check(set(sm['files'])==set(payload),'Shard payload list differs')
            for name,value in payload.items():check(sha(value)==sm['files'][name],'Shard payload checksum differs')
        records=json.loads(payload['records.json'])
        with np.load(io.BytesIO(payload['predictions.npz']),allow_pickle=False) as npz:
            arrays={key:npz[key][:8].copy() for key in npz.files}
        check(arrays['filenames'].tolist()==selected_names,'Reference image order differs')
        np.testing.assert_array_equal(arrays['targets'],selected_targets)
        images=[]; gt_points=[]
        for index,name in enumerate(selected_names):
            record=records[index]
            check(record['sample_id']==name and record['condition']==condition['condition'] and record['severity']==condition['severity'],'Record identity differs')
            with Image.open(io.BytesIO(raw_images[name])) as img:
                canvas,gt=prep.prepare_canvas(img,torch.tensor(landmarks[name],dtype=torch.float32))
            image,gt,meta=corruptions.apply_corruption(canvas,condition['implementation_kind'],condition['severity'],sample_id=name,seed=binding['corruption_seed'],gt_pixels=gt)
            check(corruptions.rgb_sha256(image)==record['final_rgb_sha256'],f'Final RGB differs: {label}/{name}')
            check(meta['parameters']==record['corruption_parameters'],'Corruption parameters differ')
            np.testing.assert_allclose(gt,record['gt_pixels'],rtol=0,atol=1e-9)
            np.testing.assert_allclose(prep.normalize_points(gt).numpy(),arrays['gt_normalized_points'][index],rtol=0,atol=1e-6)
            valid=record['detection']['status']=='ok'
            check(bool(arrays['geometry_valid'][index])==valid,'Geometry mask differs')
            if valid:
                np.testing.assert_allclose(prep.normalize_points(record['detection']['landmarks']).numpy(),arrays['predicted_normalized_points'][index],rtol=0,atol=1e-6)
            else:check(not arrays['predicted_normalized_points'][index].any(),'Invalid geometry must be zero')
            images.append(prep.normalize_canvas(image));gt_points.append(prep.normalize_points(gt))
        inputs[label]=dict(images=torch.stack(images), gt=torch.stack(gt_points),
                           predicted=torch.from_numpy(arrays['predicted_normalized_points']),
                           valid=torch.from_numpy(arrays['geometry_valid']),reference=arrays)
        print(label+': JPEG, RGB, transformed GT and geometry inputs verified',flush=True)
    environment=dict(python=platform.python_version(),torch=str(torch.__version__),
        torchvision=importlib.metadata.version('torchvision'),numpy=np.__version__,Pillow=PIL.__version__,
        device=str(device),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,cpu_threads=torch.get_num_threads())
    protocol=dict(reference_archive_sha256=ARCHIVE_SHA,reference_execution_binding_sha256=sha(binding_bytes),
        source_sha256=sha(source_code.encode()),seed=1,models=list(MODELS),conditions=list(LABELS),
        sample_selection='first eight official test IDs in partition-file order; no outcome selection',
        filenames=selected_names,attribute_names=attributes,environment=environment,
        reference_environment=binding['environment'],reference_runtime=binding['runtime'],
        model_sample_evaluations=128,precision='FP32',replay_batch_size=8,reference_batch_size=binding['batch_size'],
        logit_atol=1e-4,logit_rtol=1e-4,score_atol=1e-5,score_rtol=1e-5,
        decision_requirement='zero changed decisions under each frozen vector and fixed 0.5',
        scope='attribute-model inference with archived MTCNN inputs; detector is not rerun',
        original_shards=source_shards,training=False,threshold_fitting=False)
    # A separate session directory avoids overwriting timed reports from earlier attempts.
    import datetime
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result_root=output/stamp
    atomic(result_root/'execution_binding.json',encode(protocol))
    atomic(result_root/'source/reproduce_small_inference.py',source_code.encode())
    summaries=[];comparisons=[]; replay_arrays={}
    for model_name in MODELS:
        run_name=f'{model_name}_seed1'; root=project/'runs'/run_name
        frozen_model=binding['models'][run_name]
        config_path=root/'config.json';check(file_sha(config_path)==frozen_model['config_sha256'],'Run config SHA256 differs')
        config=json.loads(config_path.read_bytes())
        check(config['seed']==1 and config['run_name']==run_name and config['attribute_names']==attributes,'Model configuration differs')
        check(config['source_sha256']==frozen_model['source_sha256'],'Archived source binding differs')
        for relative,digest in frozen_model['source_sha256'].items():
            rp=Path(relative);check(not rp.is_absolute() and '..' not in rp.parts,'Unsafe source path')
            check(file_sha(root/'source'/rp)==digest,f'Archived source differs: {run_name}/{relative}')
        best=json.loads((root/'best.json').read_bytes())
        epoch=best['epoch'];filename=best.get('checkpoint',f'epoch_{epoch:02d}.pt')
        check(epoch==frozen_model['selected_epoch'] and filename==frozen_model['checkpoint'],'Best checkpoint changed')
        checkpoint_path=root/filename
        print('Verifying checkpoint SHA256:',run_name,flush=True)
        check(file_sha(checkpoint_path)==frozen_model['checkpoint_sha256'],'Checkpoint SHA256 differs')
        threshold_path=project/'runs/benchmark_preparation/seed1'/(run_name+'.json')
        check(file_sha(threshold_path)==frozen_model['threshold_sha256'],'Frozen threshold file differs')
        threshold_report=json.loads(threshold_path.read_bytes())
        check(threshold_report['attribute_names']==attributes and threshold_report['fit_split']=='valid','Threshold metadata differs')
        thresholds=threshold_report['thresholds']
        check(threshold_report['checkpoint_sha256']==frozen_model['checkpoint_sha256'],'Threshold/checkpoint binding differs')
        for label in LABELS:
            expected=next(item for item in report['conditions'] if item['condition']==conditions[label])
            check(expected['models'][run_name]['frozen_thresholds']['thresholds']==thresholds,'Reference thresholds differ')
        module_filename={'r18':'resnet18','r50':'resnet50','m3':'resnet18_geometry','oracle':'resnet18_geometry'}[model_name]
        module=load_module('_fvp_replay_'+run_name,root/'source/models'/(module_filename+'.py'))
        class_name={'r18':'ResNet18Attributes','r50':'ResNet50Attributes','m3':'ResNet18GeometryAttributes','oracle':'ResNet18GeometryAttributes'}[model_name]
        load_started=time.perf_counter()
        model=getattr(module,class_name)(num_attributes=24,pretrained=False).to(device)
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        check(checkpoint['epoch']==epoch and checkpoint['config']==config,'Checkpoint metadata differs')
        model.load_state_dict(checkpoint[frozen_model['state_key']],strict=True)
        del checkpoint
        model.eval()
        load_seconds=time.perf_counter()-load_started
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for label in LABELS:
            item=inputs[label];images=item['images'].to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.inference_mode():
                if model_name in ('r18','r50'):logits=model(images)
                else:
                    points=(item['predicted'] if model_name=='m3' else item['gt']).to(device)
                    valid=item['valid'].to(device) if model_name=='m3' else torch.ones(8,dtype=torch.bool,device=device)
                    logits=model(images,points,valid)
            if device.type=='cuda':torch.cuda.synchronize()
            seconds=time.perf_counter()-t
            actual=logits.cpu().numpy();reference=item['reference'][run_name+'_logits']
            check(actual.shape==(8,24),'Unexpected inference shape')
            result=compare(reference,actual,thresholds)
            summaries.append(dict(run=run_name,condition=label,samples=8,load_seconds=load_seconds,
                                  inference_seconds=seconds,**result))
            replay_arrays[run_name+'__'+label+'__actual_logits']=actual
            replay_arrays[run_name+'__'+label+'__reference_logits']=reference
            rp,ap=sigmoid(reference),sigmoid(actual)
            for i,name in enumerate(selected_names):
                for j,attribute in enumerate(attributes):
                    comparisons.append(dict(run=run_name,condition=label,filename=name,attribute=attribute,
                        target=int(selected_targets[i,j]),reference_logit=float(reference[i,j]),replayed_logit=float(actual[i,j]),
                        reference_score=float(rp[i,j]),replayed_score=float(ap[i,j]),frozen_threshold=thresholds[j],
                        reference_frozen=int(rp[i,j]>=thresholds[j]),replayed_frozen=int(ap[i,j]>=thresholds[j]),
                        reference_at_0_5=int(rp[i,j]>=.5),replayed_at_0_5=int(ap[i,j]>=.5)))
            print(f"{run_name} {label}: max logit error={result['max_absolute_logit_error']:.3g}; "
                  f"decision changes={sum(v['changed_decisions'] for v in result['decisions'].values())}; "
                  f"{'PASS' if result['replay_passed'] else 'REVIEW REQUIRED'}",flush=True)
            del logits,images
        del model
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    passed=all(s['replay_passed'] for s in summaries)
    resource_report=dict(wall_seconds=time.perf_counter()-started,
        inference_seconds=sum(s['inference_seconds'] for s in summaries),
        checkpoint_bytes=sum((project/'runs'/f'{m}_seed1'/binding['models'][f'{m}_seed1']['checkpoint']).stat().st_size for m in MODELS),
        monetary_cost=None,monetary_cost_note='Not measured automatically; use actual Colab billing or compute-unit records. No cost estimate is inferred.')
    result=dict(execution_complete=True,reproduction_passed=passed,model_sample_evaluations=128,
                attribute_comparison_rows=len(comparisons),input_reproduction_passed=True,
                detector_rerun=False,training=False,threshold_fitting=False,results=summaries,resources=resource_report,
                limitation='Small fixed subset with batch-size and hardware changes; not a rerun of the full benchmark or detector pipeline.')
    atomic(result_root/'reproduction_report.json',encode(result))
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    atomic(result_root/'attribute_comparisons.csv',buf.getvalue().encode())
    npz=io.BytesIO();np.savez_compressed(npz,filenames=np.asarray(selected_names),attribute_names=np.asarray(attributes),**replay_arrays)
    atomic(result_root/'replayed_logits.npz',npz.getvalue())
    atomic(result_root/'README.md',b'''# Small-scale inference reproduction

Seed1, four models, first eight official test images and four conditions: 128 model-sample evaluations.
Model checkpoints, archived source, frozen thresholds, JPEG bytes, final RGB and geometry inputs are verified.
MTCNN outputs are reused; this does not validate a fresh detector execution or the full 21-condition benchmark.
FP32 reference logits were generated on GPU with a different batch size; replay hardware is recorded.
Tolerances are declared in execution_binding.json before model inference. Every threshold decision is compared.
An execution-complete report may still require review if numeric tolerances fail or any decision changes.
No performance metrics or model ranking should be inferred from this small subset.
Runtime is measured; monetary cost must come from actual billing records and is not estimated here.
''')
    payloads={p.relative_to(result_root).as_posix():p.read_bytes() for p in sorted(result_root.rglob('*')) if p.is_file()}
    payloads['archive_manifest.json']=encode(dict(files={n:sha(d) for n,d in payloads.items()}))
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(payloads.items()):z.writestr(name,data)
    destination=result_root/'small_scale_reproduction_archive.zip';atomic(destination,stream.getvalue())
    print('Small-scale inference reproduction execution: COMPLETE')
    print('Reproduction status:', 'PASS' if passed else 'REVIEW REQUIRED')
    print('Archive:',destination)
    print('Archive SHA256:',sha(stream.getvalue()))
    print('Monetary cost: not automatically measured')
    return destination
