# %% Modules

import os

from argparse import ArgumentParser

from flytekit.core.pod_template import PodTemplate
from flytekitplugins.kfpytorch import (
    CleanPodPolicy,
    Elastic,
    RunPolicy,
)
from kubernetes.client.models import (
    V1Container,
    V1PodSpec,
    V1ResourceRequirements,
)

# %% Parse infrastructure arguments

_parser = ArgumentParser(add_help=False)
_parser.add_argument("--num-pods", type=int, default=1, help="Number of worker pods")
_parser.add_argument("--cpus-per-pod", type=int, default=2, help="Number of CPUs per pod")
_parser.add_argument("--memory-per-pod", type=str, default="8Gi", help="Memory per pod")
_parser.add_argument("--image", type=str, default="localhost:5001/tinymoe:latest", help="Container image")
_parser.add_argument("--timeout", type=int, default=3600, help="Task timeout in seconds")
_parser.add_argument("--wait-remove-pods", type=int, default=600, help="Seconds to wait before removing pods")
_parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries")

_args, _ = _parser.parse_known_args()

os.environ.setdefault("TINYMOE_IMAGE", _args.image)
os.environ.setdefault("TINYMOE_CPUS_PER_POD", str(_args.cpus_per_pod))
os.environ.setdefault("TINYMOE_MEMORY_PER_POD", _args.memory_per_pod)
os.environ.setdefault("TINYMOE_NUM_PODS", str(_args.num_pods))
os.environ.setdefault("TINYMOE_TIMEOUT", str(_args.timeout))
os.environ.setdefault("TINYMOE_WAIT_REMOVE_PODS", str(_args.wait_remove_pods))
os.environ.setdefault("TINYMOE_MAX_RETRIES", str(_args.max_retries))

# %% Set container image

container_image = os.environ["TINYMOE_IMAGE"]

# %% Set per-pod resources

num_cpus_per_pod = int(os.environ["TINYMOE_CPUS_PER_POD"])
memory_per_pod = os.environ["TINYMOE_MEMORY_PER_POD"]

# %% Set distributed, elastic PyTorch job

num_pods = int(os.environ["TINYMOE_NUM_PODS"])
timeout = int(os.environ["TINYMOE_TIMEOUT"])
wait_remove_pods = int(os.environ["TINYMOE_WAIT_REMOVE_PODS"])
max_retries = int(os.environ["TINYMOE_MAX_RETRIES"])

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

# %% Set resources

resources = {
    "cpu": num_cpus_per_pod,
    "memory": memory_per_pod,
}

# %% Set pod template

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

# %% End of script
