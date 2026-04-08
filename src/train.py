"""
Training code
"""

# %% Modules

from argparse import ArgumentParser

import torch
import torch.nn as nn

from torch.distributed import init_process_group, destroy_process_group
from flytekit import task, workflow
from flytekit.core.pod_template import PodTemplate
from flytekitplugins.kfpytorch import (
    CleanPodPolicy,
    Elastic,
    RunPolicy,
)
from kubernetes.client.models import (
    V1Affinity,
    V1NodeAffinity,
    V1NodeSelector,
    V1NodeSelectorTerm,
    V1NodeSelectorRequirement,
    V1Container,
    V1EnvVar,
    V1EnvVarSource,
    V1LocalObjectReference,
    V1PodSpec,
    V1ResourceRequirements,
    V1SecretKeySelector,
)

from src.args import Args
from src.models import SimpleMoE

# %% Specifications

container_image = "localhost:5001/tinymoe:latest"

num_pods = 1

num_cpus_per_pod = 1
memory_per_pod = "8Gi"
num_cpus_per_pod = 0

timeout = 60 * 60
wait_remove_pods = 600
max_retries = 3

task_config = Elastic(
    nnodes=f"{num_pods}:{num_pods}",
    nproc_per_node=num_cpus_per_pod,
    run_policy=RunPolicy(
        clean_pod_policy=CleanPodPolicy.NONE,
        ttl_seconds_after_finished=wait_remove_pods,
        backoff_limit=1,
        active_deadline_seconds=timeout,
    ),
)

resources = {
    "cpu": num_cpus_per_pod,
    "memory": memory_per_pod,
}

pod_template = PodTemplate(
    pod_spec=V1PodSpec(
        termination_grace_period_seconds=300,
        containers=[
            V1Container(
                name="primary", 
                image_pull_policy="IfNotPresent",
                resources=V1ResourceRequirements(
                    requests=resources,
                    limits=resources,
                ),
            ),
        ],
    ),
)

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
