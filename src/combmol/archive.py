"""Resolve source identity by properties, never by file position."""
import io
import tarfile
from pathlib import Path, PurePosixPath
from rdkit import Chem
from .io import digest, frozen, read, value_hash


def index_target(config, target, destination):
    destination = Path(destination)
    receipt = destination / "INDEX.json"
    if receipt.exists():
        result = read(receipt)
        archive = Path(config["dataset_root"]) / result["archive_relative"]
        if digest(archive) != result["archive_sha256"]:
            raise ValueError("Source archive changed")
        for relative, checksum in result["member_hashes"].items():
            if digest(destination / relative) != checksum:
                raise ValueError("Indexed source member changed")
        return result
    dataset = Path(config["dataset_root"])
    summary = read(dataset / "results_summary.json")
    archive_relative = summary["targets"][target]["archive"]
    archive = (dataset / archive_relative).resolve()
    if not archive.is_relative_to(dataset.resolve()):
        raise ValueError("Unsafe archive location")
    prefix = f"targets/{target}/"
    member_hashes = {}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix):]
            if ".." in PurePosixPath(relative).parts or relative.startswith("/"):
                raise ValueError("Unsafe archive member")
            wanted = relative.startswith("inputs/") or relative.endswith(
                ("molecules_all_built.sdf", "final_records.json", "config.json"))
            if not wanted:
                continue
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = stream.extractfile(member).read()
            if output.exists() and output.read_bytes() != payload:
                raise ValueError("Existing source member differs")
            output.write_bytes(payload)
            member_hashes[relative] = digest(output)
    campaign = destination / "results/unguided_w050"
    records = read(campaign / "final_records.json")
    identities = [record["node_id"] for record in records]
    if len(records) != 100 or len(set(identities)) != 100:
        raise ValueError("Expected the complete unique 100-attempt source population")
    molecules = {}
    for source in sorted((campaign / "unguided").glob("batch_*/molecules_all_built.sdf")):
        for molecule in Chem.SDMolSupplier(str(source), removeHs=True):
            if molecule is None or not molecule.HasProp("node_id"):
                raise ValueError("Source built SDF lacks a usable node_id")
            identifier = molecule.GetProp("node_id")
            if identifier in molecules:
                raise ValueError("Duplicate source node_id")
            molecules[identifier] = (molecule, source)
    expected = {record["node_id"] for record in records if record["build_success"]}
    if expected != set(molecules):
        raise ValueError("Source records and built structures disagree")
    archive_sha = digest(archive)
    candidates = []
    for record in records:
        identifier = record["node_id"]
        if identifier not in molecules:
            continue
        molecule, member = molecules[identifier]
        if molecule.HasProp("pic50_on_rescore") and abs(
                float(molecule.GetProp("pic50_on_rescore")) - record["pic50_on_rescore"]) > 1e-6:
            raise ValueError("Source score/property association differs")
        source_identity = dict(target=target, archive_sha256=archive_sha,
                               member=member.relative_to(destination).as_posix(), node_id=identifier)
        candidate = dict(identifier=target + "/" + identifier, evidence_id="source:" + value_hash(source_identity),
                         source_identity=source_identity, record=record,
                         molblock=Chem.MolToMolBlock(molecule),
                         source_member_sha256=digest(member))
        candidates.append(candidate)
    result = dict(target=target, archive_relative=archive_relative, archive_sha256=archive_sha,
                  member_hashes=member_hashes, records=records, candidates=candidates,
                  protein=str(destination / "inputs/target_protein.pdb"),
                  reference_ligand=str(destination / "inputs/target_ligand.sdf"),
                  alignment=dict(method="identity", reason="all candidates share the exact archived protein", protein_sha256=digest(destination / "inputs/target_protein.pdb")))
    return frozen(receipt, result)
