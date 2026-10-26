"""Offline synthetic checks. No pretrained downloads, real data or run writes."""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from models.rq5 import make_model
from training.rq5 import mtl_loss, write_torch
from training.rq5_shared import prepare_canvas, normalize_geometry, set_seed


def main():
    torch.set_num_threads(2)
    set_seed(0)
    image = torch.randn(2,3,224,224)
    points = torch.randn(2,10)
    valid = torch.tensor([True,False])
    target = torch.randint(0,2,(2,24)).float()
    for name in ['N0','N1','MTL']:
        model = make_model(name,pretrained=False).eval()
        with torch.no_grad():
            output = model(image,points,valid) if name=='N1' else model(image)
            logits = output[0] if name=='MTL' else output
            assert logits.shape == (2,24) and torch.isfinite(logits).all()
            if name=='N1':
                geometry = model.encode_geometry(points,valid)
                assert torch.count_nonzero(geometry[1]) == 0
                changed = points.clone(); changed[1] += 100
                torch.testing.assert_close(model(image,points,valid),model(image,changed,valid),rtol=0,atol=0)
            if name=='MTL':
                assert output[1].shape == (2,10)
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        logits,coords = model(image) if name=='MTL' else (model(image,points,valid) if name=='N1' else model(image),None)
        if name=='MTL':
            loss,bce,mse = mtl_loss(logits,coords,target,points,1.)
            for component in (bce,mse):
                gradient = torch.autograd.grad(component,model.image_encoder.conv1.weight,retain_graph=True)[0]
                assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        # Check model+optimizer continuation, not just serialization shapes.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'latest.pt'
            write_torch(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
            saved = torch.load(path,map_location='cpu',weights_only=True)
            restored = make_model(name,pretrained=False).eval()
            restored.load_state_dict(saved['model'],strict=True)
            opt = torch.optim.Adam(restored.parameters(),lr=1e-4)
            opt.load_state_dict(saved['optimizer'])
            for current,current_opt in [(model,optimizer),(restored,opt)]:
                result = current(image,points,valid) if name=='N1' else current(image)
                out = result[0] if name=='MTL' else result
                torch.nn.functional.binary_cross_entropy_with_logits(out,target).backward()
                current_opt.step(); current_opt.zero_grad(set_to_none=True)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
        print(name,'PASS; trainable parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))
    raw = np.array([[0,0],[177,0],[90,100],[0,217],[177,217]],dtype=np.float32)
    canvas,xy = prepare_canvas(Image.new('RGB',(178,218)),raw)
    assert canvas.size==(224,224)
    assert normalize_geometry(xy).shape==(10,)
    torch.testing.assert_close(normalize_geometry(xy).reshape(5,2),torch.tensor(2*xy/223-1))
    print('Synthetic checks PASS. Real cache/hash checks and CUDA training remain separate.')


if __name__=='__main__':
    main()
