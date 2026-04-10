"""
Training code
"""

# %% Modules

import io
import os
import logging

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import boto3
import torch
import torch.nn as nn
import wandb

from torch.distributed import init_process_group, destroy_process_group
from flytekit import task

from src.args import Args
from src.models import PolynomialMoE
from src.infra import container_image, task_config, pod_template, timeout, max_retries
from data.loader import create_dataloader

# %% Constants

CHECKPOINT_BUCKET = "tinymoe"
CHECKPOINT_PREFIX = "checkpoints/polynomial_moe_"

# %% Functions

def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT"),
        aws_access_key_id=os.environ.get("MINIO_ROOT_USER"),
        aws_secret_access_key=os.environ.get("MINIO_ROOT_PASSWORD"),
    )

def _load_latest_checkpoint(model, optimizer):
    s3 = _get_s3_client()

    try:
        response = s3.list_objects_v2(Bucket=CHECKPOINT_BUCKET, Prefix=CHECKPOINT_PREFIX)
    except Exception as e:
        print(f"Could not list checkpoints: {e}")
        return 0

    objects = response.get("Contents", [])

    if not objects:
        print("No checkpoint found, starting from scratch")
        return 0

    latest_key = sorted(objects, key=lambda o: o["Key"])[-1]["Key"]
    print(f"Loading checkpoint: {latest_key}")

    buf = io.BytesIO()
    s3.download_fileobj(CHECKPOINT_BUCKET, latest_key, buf)
    buf.seek(0)

    checkpoint = torch.load(buf, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"] + 1

def _save_checkpoint(model, optimizer, epoch):
    key = f"{CHECKPOINT_PREFIX}{epoch:04d}.pt"

    buf = io.BytesIO()
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }, buf)
    buf.seek(0)

    s3 = _get_s3_client()
    s3.upload_fileobj(buf, CHECKPOINT_BUCKET, key)
    print(f"Saved checkpoint: {key}")

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
    rank = torch.distributed.get_rank()

    if rank == 0:
        wandb.init(project="tinymoe", config={
            "dim": config.dim,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "lr": config.lr,
            "num_samples": config.num_samples,
            "threshold": config.threshold,
        })
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    model = PolynomialMoE(dim=config.dim).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    dataloader = create_dataloader(
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        threshold=config.threshold,
    )

    start_epoch = 0

    if rank == 0:
        start_epoch = _load_latest_checkpoint(model, optimizer)

    expert_loss_fn = nn.MSELoss()
    router_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_expert_loss = 0.0
        epoch_router_loss = 0.0
        num_batches = 0

        for points, expert_labels, projections, distances in dataloader:
            output, router_logits = model(points)

            expert_loss = expert_loss_fn(output, projections)
            router_loss = router_loss_fn(router_logits, expert_labels)
            loss = expert_loss + router_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_expert_loss += expert_loss.item()
            epoch_router_loss += router_loss.item()
            num_batches += 1

        avg_expert_loss = epoch_expert_loss / num_batches
        avg_router_loss = epoch_router_loss / num_batches

        if rank == 0:
            wandb.log({
                "epoch": epoch,
                "expert_loss": avg_expert_loss,
                "router_loss": avg_router_loss,
                "total_loss": avg_expert_loss + avg_router_loss,
            })
            print(f"Epoch {epoch}: expert_loss={avg_expert_loss:.4f} router_loss={avg_router_loss:.4f}")

            if epoch % 100 == 0:
                _save_checkpoint(model, optimizer, epoch)

    if rank == 0:
        wandb.finish()
        _save_checkpoint(model, optimizer, epoch)

    destroy_process_group()

# %% End of script
