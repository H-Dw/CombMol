"""Start one pinned campaign without exposing interactive-shell startup output."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from combmol.config import load
from combmol.io import exclusive, frozen, read, value_hash, write


def process_matches(launch):
    path = Path(f"/proc/{launch.get('pid', 0)}/stat")
    if not path.exists():
        return False
    fields = path.read_text().rsplit(")", 1)[1].split()
    return fields[0] != "Z" and fields[19] == str(launch.get("start_ticks")) and (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == launch.get("boot_id"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attempt", required=True)
    options = parser.parse_args()
    if not options.attempt.startswith("attempt_") or not options.attempt[8:].isdigit():
        raise ValueError("Use a unique attempt_NNN identifier")
    config_path = Path(options.config).resolve()
    config = load(config_path)
    root = Path(config["result_root"])
    with exclusive(root / "LAUNCH.lock"):
        for name in ("LAUNCH.json", "BACKGROUND_LAUNCH.json"):
            if (root / name).exists() and process_matches(read(root / name)):
                raise RuntimeError("The recorded campaign process is still alive")
        if subprocess.check_output(["git", "-C", str(REPOSITORY), "status", "--porcelain"], text=True).strip():
            raise ValueError("Launch requires a clean pinned checkout")
        revision = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
        run_folder = root / "runs" / options.attempt
        if run_folder.exists():
            raise ValueError("Attempt directory already exists; preserve it and use a new identifier")
        run_folder.mkdir(parents=True)
        scientific_config = {key: value for key, value in config.items() if key != "repository"}
        plan = dict(protocol=config["protocol"], configuration=scientific_config,
                    configuration_hash=value_hash(scientific_config), initial_revision=revision,
                    scientific_attempt_upper_bound=800, attempts_per_condition=25,
                    credential_policy="process environment only; no credential values in receipts")
        if (root / "PLAN.json").exists():
            if read(root / "PLAN.json")["configuration_hash"] != plan["configuration_hash"]:
                raise ValueError("Scientific plan changed during recovery")
        else:
            frozen(root / "PLAN.json", plan)
        command = "exec bash " + " ".join(shlex.quote(str(value)) for value in (
            REPOSITORY / "scripts/run_campaign.sh", config_path, options.attempt))
        command += " >" + shlex.quote(str(run_folder / "campaign.log")) + " 2>&1"
        # The shell loads the existing API/runtime environment. Its startup text is discarded.
        process = subprocess.Popen(["bash", "-ic", command], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        launch = dict(pid=process.pid, start_ticks=fields[19],
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(), time=time.time(),
            revision=revision, repository=str(REPOSITORY), attempt=options.attempt,
            log=str(run_folder / "campaign.log"), kind="background wrapper; exec preserves process identity")
        frozen(run_folder / "BACKGROUND_LAUNCH.json", launch)
        write(root / "BACKGROUND_LAUNCH.json", launch)
        write(root / "LAUNCH.json", launch)
        print(json.dumps(launch))


if __name__ == "__main__":
    main()
