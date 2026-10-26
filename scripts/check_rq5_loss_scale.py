"""Measure training-only BCE/MSE scales; never select lambda or update weights."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.rq5 import make_model
from datasets.celeba_rq5 import CelebARQ5
from training.rq5 import load_config, make_loader
from training.rq5_shared import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True)
    parser.add_argument('--images',required=True)
    parser.add_argument('--config',default='configs/rq5_training_v1.json')
    args = parser.parse_args()
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(0)
    model = make_model('MTL',pretrained=True).to(device).train()
    dataset = CelebARQ5(args.project,args.images,'train','MTL')
    loader = make_loader(dataset,config,True,seed=1)
    scales = []
    with torch.no_grad():
        for i,batch in enumerate(loader):
            logits,xy = model(batch['image'].to(device))
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits,batch['target'].to(device))
            mse = torch.nn.functional.mse_loss(xy,batch['gt'].to(device))
            scales.append([float(bce),float(mse)])
            if i==3:
                break
    print('Training-only scale check; seed=0, 4 batches, no optimizer updates')
    print('BCE/MSE by batch:',scales)
    mean = np.mean(scales,axis=0)
    print('Mean BCE/MSE:',mean.tolist())
    print('Equal-loss-scale reference BCE/MSE:',float(mean[0]/mean[1]))
    print('Review and record ONE lambda plus lambda_basis in config before any formal run.')
    print('This ratio is a scale diagnostic, not evidence of the best lambda.')


if __name__=='__main__':
    main()
