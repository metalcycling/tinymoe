# %% Modules

import torch
import torch.nn as nn

# %% Classes

class DummyExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        return self.net(x)

class SimpleMoE(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([DummyExpert(dim) for _ in range(num_experts)])

    def forward(self, x):
        logits = self.router(x)
        weights = torch.softmax(logits, dim=-1)
        best_expert = torch.argmax(weights, dim=-1)

        out = torch.zeros_like(x)

        for eid, expert in enumerate(self.experts):
            mask = (best_expert == eid)

            if mask.any():
                out[mask] = expert(x[mask])

        return out

# %% End of program
