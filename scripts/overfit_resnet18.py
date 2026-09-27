"""Check whether ResNet18 can memorize eight training images."""
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from datasets.celeba import CelebAAttributes
from models.resnet18 import ResNet18Attributes


def main():
    seed = 42
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    dataset = CelebAAttributes("train")
    indices = random.Random(seed).sample(range(len(dataset)), 8)
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample[0] for sample in samples])
    targets = torch.stack([sample[1] for sample in samples])
    names = [sample[3] for sample in samples]

    model = ResNet18Attributes(len(dataset.attribute_names), pretrained=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = []
    started = time.perf_counter()
    passed = False

    def evaluate():
        model.eval()
        with torch.inference_mode():
            logits = model(images)
            loss = criterion(logits, targets).item()
            predictions = (logits >= 0).float()
            exact = (predictions == targets).all(dim=1).float().mean().item()
        return loss, exact

    initial_loss, initial_exact = evaluate()
    print("Samples:", names, flush=True)
    print(f"Initial eval loss: {initial_loss:.6f}", flush=True)

    for step in range(1, 101):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        loss.backward()

        if step == 1:
            for name, parameter in [
                ("backbone", model.network.conv1.weight),
                ("head", model.network.fc.weight),
            ]:
                gradient = parameter.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError(f"Missing or invalid gradient: {name}")
                norm = gradient.norm().item()
                if norm == 0:
                    raise RuntimeError(f"Zero gradient: {name}")
                print(f"{name} gradient norm: {norm:.6f}", flush=True)

        optimizer.step()

        if step == 1 or step % 10 == 0:
            eval_loss, exact = evaluate()
            history.append({
                "step": step,
                "train_loss_before_update": loss.item(),
                "eval_loss": eval_loss,
                "exact_match": exact,
            })
            print(
                f"Step {step:03d} | train loss {loss.item():.6f} | "
                f"eval loss {eval_loss:.6f} | exact match {exact:.1%}",
                flush=True,
            )
            if eval_loss < 0.05 and exact == 1.0:
                passed = True
                break

    report = {
        "purpose": "Training-pipeline check, not generalization evaluation",
        "seed": seed,
        "device": "cpu",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "attribute_names": dataset.attribute_names,
        "sample_ids": names,
        "initial_eval_loss": initial_loss,
        "initial_exact_match": initial_exact,
        "history": history,
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = Path("outputs/logs/resnet18_overfit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Result:", "PASS" if passed else "NEEDS REVIEW", flush=True)
    print(f"Elapsed: {report['elapsed_seconds']:.1f} seconds", flush=True)
    print("Saved:", path, flush=True)


if __name__ == "__main__":
    main()
