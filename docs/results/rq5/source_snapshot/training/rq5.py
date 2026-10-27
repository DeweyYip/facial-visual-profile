"""RQ5 training and inference. Importing this module starts no experiment."""

import argparse
import copy
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from torch import nn
from torch.utils.data import DataLoader

from models.rq5 import make_model
from datasets.celeba_rq5 import CelebARQ5, ATTRIBUTES
from training.rq5_shared import set_seed, seed_worker, compute_metrics, fit_thresholds


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_torch(path, value):
    temp = path.with_name(path.name + '.partial')
    torch.save(value, temp)
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    expected = dict(study='rq5_training_v1', seeds=[0,1,2], models=['N0','N1','MTL'],
        epochs=10,min_epochs=4,patience=2,batch_size=64,eval_batch_size=128,
        num_workers=2,learning_rate=1e-4,weight_decay=0.,precision='FP32',
        augmentation=False,mask_rgb=False,pretrained='ResNet18_Weights.IMAGENET1K_V1',
        checkpoint_selection='maximum clean-validation mAP; earliest exact tie')
    for key,value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Frozen RQ5 setting differs: {key}')
    return config


def make_loader(dataset, config, training=False, seed=0):
    return DataLoader(dataset,
        batch_size=config['batch_size'] if training else config['eval_batch_size'],
        shuffle=training, num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed), persistent_workers=False)


def forward(model, name, batch, device):
    images = batch['image'].to(device)
    if name == 'N1':
        return model(images, batch['points'].to(device), batch['valid'].to(device)), None
    value = model(images)
    return value if name == 'MTL' else (value, None)


def mtl_loss(logits, coordinates, target, gt, weight):
    if not isinstance(weight, (float,int)) or not np.isfinite(weight) or weight <= 0:
        raise ValueError('Freeze a positive finite MTL lambda before training')
    attribute = nn.functional.binary_cross_entropy_with_logits(logits, target)
    landmark = nn.functional.mse_loss(coordinates, gt)
    return attribute + weight*landmark, attribute, landmark


@torch.inference_mode()
def predict(model, name, dataset, config, device):
    model.eval()
    labels, probabilities, nme = [], [], []
    for batch_index,batch in enumerate(make_loader(dataset, config)):
        if batch_index % 80 == 0:
            print('  inference batches:',batch_index,'/',len(dataset)//config['eval_batch_size']+1,flush=True)
        logits, coordinates = forward(model, name, batch, device)
        if logits.shape != (len(batch['image']),24) or not torch.isfinite(logits).all():
            raise ValueError('Invalid attribute outputs')
        labels.append(batch['target'].numpy())
        probabilities.append(logits.sigmoid().cpu().numpy())
        if coordinates is not None:
            xy = coordinates.cpu().reshape(-1,5,2)
            gt = batch['gt'].reshape(-1,5,2)
            eyes = torch.linalg.vector_norm(gt[:,0]-gt[:,1],dim=1)
            if not torch.isfinite(xy).all() or not (eyes>0).all():
                raise ValueError('Invalid landmark diagnostic')
            nme.extend((torch.linalg.vector_norm(xy-gt,dim=2).mean(1)/eyes).tolist())
    return np.concatenate(labels), np.concatenate(probabilities), nme


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ['models/rq5.py','datasets/celeba_rq5.py','training/rq5.py',
             'training/rq5_shared.py','datasets/landmark_cache.py','datasets/corruptions.py']
    return {path:sha256(root/path) for path in paths}


def bind_run(folder, config, name, seed):
    environment = dict(python=platform.python_version(),torch=torch.__version__,
        torchvision=torchvision.__version__,Pillow=pillow_version,cuda=torch.version.cuda,
        numpy=np.__version__,sklearn=__import__('sklearn').__version__,gpu=torch.cuda.get_device_name(0))
    binding = dict(config=config,model=name,seed=seed,source_hashes=source_hashes(),
                   environment=environment,attributes=ATTRIBUTES)
    path = folder/'config.json'
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError('Existing run config/source/environment differs; do not mix runs')
    else:
        write_json(path,binding)


def train_one(project, image_root, name, seed, config, device):
    if name == 'MTL' and (config['mtl_lambda'] is None or not config['lambda_basis']):
        raise ValueError('MTL lambda and its training-only scale basis are not frozen')
    folder = Path(project)/'runs'/config['study']/f'{name}_seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    bind_run(folder,config,name,seed)
    if (folder/'train_complete.json').exists():
        done = json.loads((folder/'train_complete.json').read_text())
        assert (folder/'latest.pt').exists() and (folder/'best_model.pt').exists()
        selected = torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)
        assert selected['exp']==name and selected['seed']==seed
        assert selected['epoch']==done['best']['epoch']
        print('SKIP completed training:',folder.name,'epochs:',done['epochs'],flush=True)
        return
    # A first-epoch disconnect can leave sidecars before latest.pt commits.
    # Preserve these uncommitted files, then repeat epoch 1 from its original seed.
    if not (folder/'latest.pt').exists():
        orphans = [folder/f for f in ('history.json','best.json','best_model.pt') if (folder/f).exists()]
        if orphans:
            import time
            recovery = folder/'uncommitted_recovery'/str(time.time_ns())
            recovery.mkdir(parents=True)
            for orphan in orphans:
                orphan.replace(recovery/orphan.name)
            print('Preserved uncommitted first-epoch sidecars:',recovery,flush=True)
    train = CelebARQ5(project,image_root,'train',name)
    valid = CelebARQ5(project,image_root,'valid',name)
    set_seed(seed)
    model = make_model(name,pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=0.)
    latest = folder/'latest.pt'
    history, best, best_state, start = [], None, None, 1
    if latest.exists():
        saved = torch.load(latest,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        history,best = saved['history'],saved['best']
        print('RESUME:',folder.name,'last complete epoch:',saved['epoch'],flush=True)
        best_state = saved['best_state']
        # latest.pt is the epoch commit point. Repair sidecars from its snapshot.
        best_path = folder/'best_model.pt'
        existing = torch.load(best_path,map_location='cpu',weights_only=True) if best_path.exists() else None
        if existing is None or existing['epoch'] != best['epoch']:
            assert not (folder/'clean_val_thresholds.json').exists(), 'Do not replace a threshold-bound checkpoint'
            write_torch(best_path,dict(model=best_state,exp=name,seed=seed,epoch=best['epoch']))
        write_json(folder/'best.json',best)
        write_json(folder/'history.json',history)
        start = saved['epoch']+1
        if [r['epoch'] for r in history] != list(range(1,start)):
            raise ValueError('Checkpoint history mismatch')
    elif any((folder/f).exists() for f in ('history.json','best_model.pt','train_complete.json')):
        raise RuntimeError('Partial run without latest.pt; manual recovery required')
    if (folder/'train_complete.json').exists():
        done = json.loads((folder/'train_complete.json').read_text())
        if done['epochs'] != start-1 or not (folder/'best_model.pt').exists():
            raise ValueError('Completion marker mismatch')
        return
    if history and history[-1]['epoch'] >= config['min_epochs'] and history[-1]['epoch']-best['epoch'] >= config['patience']:
        write_json(folder/'train_complete.json',dict(epochs=history[-1]['epoch'],best=best))
        return
    for epoch in range(start,config['epochs']+1):
        set_seed(seed+epoch)
        model.train()
        totals = np.zeros(3); seen = 0
        print('TRAIN:',folder.name,'epoch',epoch,'/',config['epochs'],flush=True)
        for batch_index,batch in enumerate(make_loader(train,config,True,seed+epoch)):
            if batch_index % 250 == 0:
                print('  train batches:',batch_index,'samples:',seen,'/',len(train),flush=True)
            optimizer.zero_grad(set_to_none=True)
            logits,coords = forward(model,name,batch,device)
            target = batch['target'].to(device)
            if name == 'MTL':
                loss,bce,mse = mtl_loss(logits,coords,target,batch['gt'].to(device),config['mtl_lambda'])
            else:
                loss = bce = nn.functional.binary_cross_entropy_with_logits(logits,target)
                mse = loss.new_zeros(())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            count = len(target); seen += count
            totals += np.array([loss.item(),bce.item(),mse.item()])*count
        truth,prob,_ = predict(model,name,valid,config,device)
        metrics = compute_metrics(truth,prob)
        row = dict(epoch=epoch,training_loss=float(totals[0]/seen),
            attribute_BCE=float(totals[1]/seen),landmark_MSE=float(totals[2]/seen),
            clean_validation_mAP=metrics['mAP'],clean_validation_F1_at_05=metrics['macro_f1'])
        history.append(row)
        if best is None or metrics['mAP'] > best['mAP']:
            best = dict(epoch=epoch,mAP=metrics['mAP'])
            best_state = copy.deepcopy(model.state_dict())
            write_torch(folder/'best_model.pt',dict(model=best_state,exp=name,seed=seed,epoch=epoch))
        write_json(folder/'history.json',history)
        write_json(folder/'best.json',best)
        write_torch(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,history=history,best=best,best_state=best_state))
        print(name,seed,row,flush=True)
        if epoch >= config['min_epochs'] and epoch-best['epoch'] >= config['patience']:
            break
    write_json(folder/'train_complete.json',dict(epochs=history[-1]['epoch'],best=best))


def load_best(project,name,seed,config,device):
    folder = Path(project)/'runs'/config['study']/f'{name}_seed{seed}'
    if not (folder/'train_complete.json').exists():
        raise ValueError(f'Training incomplete: {folder}')
    saved = torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)
    model = make_model(name,pretrained=False).to(device)
    model.load_state_dict(saved['model'],strict=True)
    return model,folder


def export_thresholds(project,image_root,name,seed,config,device):
    model,folder = load_best(project,name,seed,config,device)
    path = folder/'clean_val_thresholds.json'
    checkpoint_sha = sha256(folder/'best_model.pt')
    if path.exists():
        if json.loads(path.read_text())['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('Threshold/checkpoint binding changed')
        return
    truth,prob,_ = predict(model,name,CelebARQ5(project,image_root,'valid',name),config,device)
    write_json(path,dict(thresholds=fit_thresholds(truth,prob).tolist(),attribute_order=ATTRIBUTES,
                         checkpoint_sha256=checkpoint_sha))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True)
    parser.add_argument('--images',required=True)
    parser.add_argument('--config',default='configs/rq5_training_v1.json')
    parser.add_argument('--model',choices=['N0','N1','MTL'],required=True)
    parser.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Formal training requires the recorded CUDA environment')
    config = load_config(args.config)
    train_one(args.project,args.images,args.model,args.seed,config,'cuda')
    export_thresholds(args.project,args.images,args.model,args.seed,config,'cuda')


if __name__ == '__main__':
    main()
