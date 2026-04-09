"""
Training code
"""

# %% Modules

import logging

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
import torch.nn as nn

from torch.distributed import init_process_group, destroy_process_group
from flytekit import task

from src.args import Args
from src.models import SimpleMoE
from src.infra import container_image, task_config, pod_template, timeout, max_retries

# %% Function

@task(
    task_config=task_config,
    container_image=container_image,
    pod_template=pod_template,
    shared_memory=True,
    timeout=timeout,
    retries=max_retries,
)
def train(config: Args) -> None:
    """
    Training code
    """
    init_process_group(backend="gloo")

    model = SimpleMoE(dim=config.dim).to("cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for step in range(config.num_epochs):
        data = torch.randn(config.batch_size, config.dim)
        output = model(data)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step} completed on Rank {torch.distributed.get_rank()}")

    destroy_process_group()

# %% End of script
