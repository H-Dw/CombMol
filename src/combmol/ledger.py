"""Write-ahead batch reservation: started slots cannot become pending again."""
from pathlib import Path
from .io import digest, read, write


class AttemptLedger:
    def __init__(self, path, condition_hash, seeds, replicates=5):
        self.path = Path(path)
        self.binding = dict(condition_hash=condition_hash, seeds=seeds, replicates=replicates)
        if self.path.exists():
            self.data = read(self.path)
            if self.data["binding"] != self.binding:
                raise ValueError("Attempt ledger identity differs")
        else:
            self.data = dict(binding=self.binding, slots={
                f"{seed}:{replicate}": dict(seed=seed, replicate=replicate, state="pending")
                for seed in seeds for replicate in range(replicates)})
            write(self.path, self.data)

    def pending(self, seed):
        return [key for key, slot in self.data["slots"].items()
                if slot["seed"] == seed and slot["state"] == "pending"]

    def start(self, keys, initialization_hash, revision):
        if not keys or any(self.data["slots"][key]["state"] != "pending" for key in keys):
            raise ValueError("Cannot reserve a non-pending slot")
        for key in keys:
            self.data["slots"][key].update(state="started", initialization_hash=initialization_hash, revision=revision)
        write(self.path, self.data)

    def finish(self, key, **artifacts):
        if self.data["slots"][key]["state"] != "started":
            raise ValueError("Only a started slot can finish")
        self.data["slots"][key].update(state="complete", **artifacts)
        write(self.path, self.data)

    def spent(self):
        return sum(slot["state"] != "pending" for slot in self.data["slots"].values())

    def reconcile_saved(self, seed, rows, terminal_file):
        expected = {(seed, replicate) for replicate in range(self.binding["replicates"])}
        actual = [(row["seed"], row["replicate"]) for row in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Saved terminal batch identities are incomplete or duplicated")
        checksum = digest(terminal_file)
        keys = [f"{seed}:{replicate}" for replicate in range(self.binding["replicates"])]
        if any(self.data["slots"][key]["state"] == "pending" for key in keys):
            raise ValueError("Saved terminals have no write-ahead reservation")
        for key in keys:
            slot = self.data["slots"][key]
            if slot["state"] == "complete" and slot["terminal_sha256"] != checksum:
                raise ValueError("Saved terminal tensors changed")
        recovered = []
        for key in keys:
            if self.data["slots"][key]["state"] == "started":
                self.finish(key, terminal_file=str(terminal_file), terminal_sha256=checksum)
                recovered.append(key)
        return recovered

    def fail_started(self, reason):
        for slot in self.data["slots"].values():
            if slot["state"] == "started":
                slot.update(state="failed", reason=reason)
        write(self.path, self.data)
