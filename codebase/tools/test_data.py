import os
import yaml
from src.data.build import build_dataloaders

def main():
    with open("configs/model_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    train_loader, _ = build_dataloaders(cfg)
    batch = next(iter(train_loader))
    print(batch["rgb"].shape)

    dv = batch["depth"][batch["valid"]]
    print("depth valid min/max:", float(dv.min()), float(dv.max()))

if __name__ == "__main__":
    main()