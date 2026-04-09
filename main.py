# %% Modules

import os
import sys
import subprocess
import uuid
import json

from flytekit import workflow
from flytekit.core.options import Options
from flytekit.models.common import Labels

from argparse import ArgumentParser

from src.args import Args
from src.train import *

# %% Main function

@workflow(
    default_options=Options(
        labels=Labels(
            {
                "user": os.environ.get("USER", "unknown"),
            }
        ),
    ),
)
def main(config: Args) -> None:
    """
    Main function
    """
    train(config)

# %% Main program

if __name__ == "__main__":
    # Model arguments

    from src.infra import _parser as infra_parser

    model_parser = ArgumentParser(description="TinyMoE trainer", parents=[infra_parser])
    model_parser.add_argument("-d", "--dim", type=int, default=128, help="Dimension of the embedding")
    model_parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for the trainer")
    model_parser.add_argument("-n", "--num-epochs", type=int, default=10, help="Number of epochs to train")
    model_parser.add_argument("-lr", "--lr", type=float, default=1.0e-04, help="Learning rate for the trainer")

    model_args, _ = model_parser.parse_known_args()

    # Convert model arguments to JSON

    config = json.dumps(vars(model_args))

    # Run pyflyte

    execution_id = f"tinymoe-{str(uuid.uuid4())[:4]}"

    cmd = [
        "pyflyte",
        "--verbose",
        "run",
        "--name",
        f"{execution_id}",
        "--remote",
        "main.py",
        "main",
        "--config",
        f"{config}",
    ]

    subprocess.run(cmd)

# %% End of script
