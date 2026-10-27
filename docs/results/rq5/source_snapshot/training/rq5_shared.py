"""Frozen RQ4 helpers, extracted without executing notebook cells."""
import hashlib
import io
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import average_precision_score, f1_score
SIZE=224

MEAN=(0.485,0.456,0.406)

STD=(0.229,0.224,0.225)

FAMILIES=('blur','brightness','rotation','occlusion','jpeg')

LEVELS={
    'blur':(1,2,3,4), 'brightness':(.8,.6,.4,.2),
    'rotation':(5,10,15,20), 'occlusion':(.10,.20,.30,.40),
    'jpeg':(75,50,25,10)
}

def prepare_canvas(image, raw_points):
    width,height=image.size
    scale=SIZE/max(width,height)
    new_width,new_height=round(width*scale),round(height*scale)
    left,top=(SIZE-new_width)//2,(SIZE-new_height)//2
    resized=image.convert('RGB').resize((new_width,new_height),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(SIZE,SIZE),(0,0,0))
    canvas.paste(resized,(left,top))
    points=np.array(raw_points,dtype=np.float32).reshape(5,2).copy()
    points[:,0]=(points[:,0]+0.5)*new_width/width-0.5+left
    points[:,1]=(points[:,1]+0.5)*new_height/height-0.5+top
    return canvas,points

def direction_for(filename):
    payload=f'day14_draft_v1|0|{filename}|rotation'.encode()
    return +1 if (hashlib.sha256(payload).digest()[0] & 1) else -1

def transform_corruption(canvas, xy, family=None, severity=0, filename=''):
    if family is None or severity==0: return canvas, xy.copy()
    assert family in FAMILIES and severity in (1,2,3,4)
    value=LEVELS[family][severity-1]
    result=xy.copy()
    if family=='blur':
        canvas=canvas.filter(ImageFilter.GaussianBlur(radius=value))
    elif family=='brightness':
        canvas=ImageEnhance.Brightness(canvas).enhance(value)
    elif family=='rotation':
        degrees=direction_for(filename)*value
        canvas=canvas.rotate(degrees,resample=Image.Resampling.BILINEAR,expand=False,center=(112.0,112.0),fillcolor=(0,0,0))
        radians=np.deg2rad(degrees)
        c,s=np.cos(radians),np.sin(radians)
        offset=result-np.array([111.5,111.5],dtype=np.float32)
        result=np.stack([111.5+c*offset[:,0]+s*offset[:,1],
                         111.5-s*offset[:,0]+c*offset[:,1]],axis=1).astype(np.float32)
    elif family=='occlusion':
        side=int(math.floor(SIZE*value+0.5))
        x0=112-side//2;y0=168-side//2
        canvas=canvas.copy()
        canvas.paste((0,0,0),(x0,y0,x0+side,y0+side))
    elif family=='jpeg':
        stream=io.BytesIO()
        canvas.save(stream,format='JPEG',quality=value,subsampling=2,optimize=False,progressive=False)
        stream.seek(0)
        with Image.open(stream) as pic:
            canvas=pic.convert('RGB')
    return canvas, result

def normalize_geometry(points):
    return torch.as_tensor((2*np.asarray(points,dtype=np.float32)/(SIZE-1)-1).reshape(-1).copy(),dtype=torch.float32)

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def seed_worker(worker_id):
    w=torch.initial_seed()%(2**32)
    np.random.seed(w);random.seed(w)

def compute_metrics(y,prob,thresholds=None):
    if thresholds is None:thresholds=np.full(24,.5,dtype=np.float64)
    aps=[];fs=[]
    for a in range(24):
        assert np.unique(y[:,a]).size==2, f'Class missing for attribute {ATTRIBUTES[a]}'
        aps.append(float(average_precision_score(y[:,a],prob[:,a])))
        fs.append(float(f1_score(y[:,a],prob[:,a]>=thresholds[a],zero_division=0)))
    return {'mAP':float(np.mean(aps)),'macro_f1':float(np.mean(fs)),
            'AP_by_attr':aps,'F1_by_attr':fs,'n':len(y)}

def fit_thresholds(y,prob):
    grid=np.arange(5,96,dtype=np.float64)/100
    threshold=[]
    for a in range(24):
        gold=y[:,a].astype(bool)
        preds=prob[:,a,None]>=grid[None,:]
        tp=(preds & gold[:,None]).sum(axis=0)
        fp=(preds & ~gold[:,None]).sum(axis=0)
        fn=((~preds) & gold[:,None]).sum(axis=0)
        den=2*tp+fp+fn
        score=np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den>0)
        best=np.flatnonzero(np.isclose(score,score.max(),rtol=0,atol=1e-12))
        ix=sorted(best,key=lambda i:(abs(grid[i]-.5),grid[i]))[0]
        threshold.append(float(grid[ix]))
    return np.array(threshold,dtype=np.float64)
