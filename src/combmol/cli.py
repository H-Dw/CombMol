import argparse
import json
import os
import threading
import time
import traceback
from pathlib import Path
from .config import load
from .io import process_identity, read, write


def inspect(root):
    root = Path(root)
    result = {}
    for name in ("LAUNCH", "STATUS", "HEARTBEAT", "CPU_PREFLIGHT_EXIT", "FINISHED", "SUITE_FAILED", "ONLINE_PROTOCOL_AUDIT"):
        path = root / (name + ".json")
        if path.exists():
            result[name] = read(path)
    launch = result.get("LAUNCH", {})
    if launch.get("pid") and Path("/proc").exists():
        process = Path(f"/proc/{launch['pid']}/stat")
        live = process.exists()
        if live:
            fields = process.read_text().rsplit(")", 1)[1].split()
            live = fields[0] != "Z" and fields[19] == str(launch["start_ticks"]) and (
                Path("/proc/sys/kernel/random/boot_id").read_text().strip() == launch["boot_id"])
        result["verified_process_alive"] = live
    result["ledger_spent"] = sum(sum(slot["state"] != "pending" for slot in read(path)["slots"].values())
                                for path in root.glob("targets/*/fusions/*/ATTEMPTS.json"))
    result["conditions"] = []
    for path in sorted(root.glob("targets/*/fusions/*/SUMMARY.json")):
        result["conditions"].append(dict(path=str(path.relative_to(root)), **read(path)))
    return result


def main():
    parser = argparse.ArgumentParser(description="CombMol native interaction-fragment inpainting")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--attempt", default="attempt_001")
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--root", required=True)
    check_parser = commands.add_parser("preflight")
    check_parser.add_argument("--config", required=True)
    options = parser.parse_args()
    if options.command == "inspect":
        print(json.dumps(inspect(options.root), ensure_ascii=False))
        return
    config = load(options.config)
    if config.get("plip_dependencies"):
        import sys
        sys.path.append(config["plip_dependencies"])
    if options.command == "preflight":
        from .native import verify_sources
        verify_sources(config)
        from plip.structure.preparation import PDBComplex
        from posebusters import PoseBusters
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Accelerator is unavailable")
        print(json.dumps(dict(status="passed", torch=torch.__version__, accelerator_count=torch.cuda.device_count(),
                              protocol=config["protocol"], checkpoint_sha256=config["checkpoint_sha256"])))
        return
    from .campaign import run
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            write(Path(config["result_root"]) / "HEARTBEAT.json", dict(**process_identity(), attempt=options.attempt))
            stop.wait(30)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        result = run(config, options.attempt)
        print(json.dumps(result, ensure_ascii=False))
        if not result["all_success"]:
            raise SystemExit(1)
    except Exception as error:
        failure = dict(error_type=type(error).__name__, message=str(error)[:4000], traceback=traceback.format_exc(),
                       attempt=options.attempt, time=time.time())
        write(Path(config["result_root"]) / "runs" / options.attempt / "SUITE_FAILED.json", failure)
        write(Path(config["result_root"]) / "SUITE_FAILED.json", failure)
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
