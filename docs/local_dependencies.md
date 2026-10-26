# Local dependency profiles

Use Python 3.10 for the historical local profile. `requirements-local.txt` retains the existing torch 2.2.2 / torchvision 0.17.2 base; it is not the Colab training environment.

- Base: `python -m pip install -r requirements-local.txt`
- Attribute metrics: `python -m pip install -r requirements-metrics.txt`
- MTCNN and metrics: `python -m pip install -r requirements-detector.txt`
- Check the installed environment: `python -m pip check`

The detector profile adds facenet-pytorch 2.6.0; the metrics profile adds scikit-learn 1.6.1. Profiles are cumulative. Historical Colab runs use separately recorded software versions; do not overwrite archived configs or treat a local installation as exact training reproduction. CUDA wheels must match the intended platform. This documentation update does not certify a fresh installation or rerun experiments.
