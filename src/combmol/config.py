from pathlib import Path
from .io import read

PROTOCOL = "combmol-inpaint-v1"
SEEDS = [42, 100045, 200048, 300051, 400054]
UPSTREAM = "739a33c2aab74d074dd24605b9a2e34dc809ae9c"
CHECKPOINT = "f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0"


def load(path):
    config = read(path)
    expected = dict(protocol=PROTOCOL, seeds=SEEDS, replicates=5, max_parents=8,
                    max_fusions=16, batch_size=50, integration_steps=100,
                    apply_guidance=False, final_inpaint=False, model="deepseek-flash",
                    checkpoint_sha256=CHECKPOINT, upstream_commit=UPSTREAM)
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"Frozen protocol setting differs: {name}")
    if config["targets"] != ["6pyy_H7S_A_402", "3w54_RNB_A_504"]:
        raise ValueError("Unexpected target population")
    if any("password" in name.lower() or "api_key" in name.lower() for name in config):
        raise ValueError("Credentials cannot be configuration values")
    config["repository"] = str(Path(__file__).resolve().parents[2])
    return config
