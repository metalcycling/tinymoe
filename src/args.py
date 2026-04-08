# %% Modules

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from argparse import ArgumentParser

# %% Classes

@dataclass_json
@dataclass
class Args:
    dim: int
    batch_size: int
    num_epochs: int
    lr: float

# %% End of script
