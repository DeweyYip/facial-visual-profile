# Facial Visual Profile

Research question:
Does an explicit lightweight 5-point geometry branch improve
robustness beyond RGB model depth?

Models: ResNet18, ResNet50, Dual-Predicted.
Oracle geometry is a separate diagnostic condition.

Local Mac: development, small-batch checks, visualization and demo.
Cloud GPU: full training and large-scale evaluation.

Local setup:
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-local.txt

Attribute order: configs/attributes.yaml
