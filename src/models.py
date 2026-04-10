# %% Modules

import torch
import torch.nn as nn

# %% Classes

class PolynomialExpert(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)

class PolynomialMoE(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        assert dim == 2, f"PolynomialMoE expects dim=2 (2D points), got {dim}"
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([PolynomialExpert(dim) for _ in range(num_experts)])

    def forward(self, x):
        router_logits = self.router(x)
        best_expert = torch.argmax(router_logits, dim=-1)

        out = torch.zeros_like(x)

        for eid, expert in enumerate(self.experts):
            mask = (best_expert == eid)

            if mask.any():
                out[mask] = expert(x[mask])

        return out, router_logits

# %% End of program
