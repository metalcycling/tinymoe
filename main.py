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
    # %% Command line arguments

    parser = ArgumentParser(description="TinyMoE trainer")
    parser.add_argument("-d", "--dim", type=int, default=128, help="Dimension of the embedding")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for the trainer")
    parser.add_argument("-n", "--num-epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("-lr", "--lr", type=float, default=1.0e-04, help="Learning rate for the trainer")

    args = parser.parse_args()

    # %% Convert arguments to JSON

    config = json.dumps(vars(args))

    # %% Run pyflyte

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

    #main(args)

# %% End of script
