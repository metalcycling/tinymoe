# %% Modules

from dataclasses import dataclass
from dataclasses_json import dataclass_json

# %% Classes

@dataclass_json
@dataclass
class Args:
    dim: int
    batch_size: int
    num_epochs: int
    lr: float
    num_samples: int = 2000
    threshold: float = 0.5

# %% End of script
