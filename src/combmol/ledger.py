"""Write-ahead batch reservation: started slots cannot become pending again."""
from pathlib import Path
from .io import frozen, read, write, value_hash


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

    def fail_started(self, reason):
        for slot in self.data["slots"].values():
            if slot["state"] == "started":
                slot.update(state="failed", reason=reason)
        write(self.path, self.data)
