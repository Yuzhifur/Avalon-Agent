# seq_model/calibrate.py
#
# Temperature scaling of the 15-way pair logits on validation data (reuses
# the temperature_scaling.py pattern: single scalar T fitted by LBFGS on NLL).

import torch
import torch.nn as nn


def fit_temperature(logits, labels, max_iter=50):
    """logits: [N, 15], labels: [N] long -> fitted temperature (float)."""
    temperature = nn.Parameter(torch.ones(1))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    before = criterion(logits, labels).item()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    after = criterion(logits / temperature.detach(), labels).item()
    print(f"Temperature scaling: T={temperature.item():.4f} "
          f"NLL {before:.4f} -> {after:.4f}")
    return float(temperature.item())
