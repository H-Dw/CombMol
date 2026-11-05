import csv
import math
from pathlib import Path
import numpy
from rdkit import Chem
from rdkit.Chem import QED
from .geometry import clashes, frame_rmsd, protein_atoms
from .io import read, write, frozen, value_hash


def molecular_properties(molecule):
    from rdkit.Contrib.SA_Score import sascorer
    return dict(SAScore=float(sascorer.calculateScore(molecule)), QEDScore=float(QED.qed(molecule)),
                heavy_atoms=molecule.GetNumHeavyAtoms(), formal_charge=sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))


def parent_profile_job(task):
    from .interactions import plip_profile, quality
    molecule = Chem.MolFromMolBlock(task["molblock"], removeHs=True)
    try:
        profile = plip_profile(molecule, task["protein"], Path(task["folder"]) / "plip")
        for position, contact in enumerate(profile["interactions"]):
            contact["evidence_id"] = "interaction:" + value_hash(dict(source=task["source_evidence"], position=position, contact=contact))
        write(Path(task["folder"]) / "PROFILE.json", profile)
        diagnostics = quality(molecule, task["protein"])
        write(Path(task["folder"]) / "QUALITY.json", diagnostics)
        return dict(identifier=task["identifier"], profile=profile, diagnostics=diagnostics)
    except Exception as error:
        result = dict(identifier=task["identifier"], profile_error=dict(type=type(error).__name__, message=str(error)[:2000]))
        write(Path(task["folder"]) / "PROFILE_FAILED.json", result)
        return result


def terminal_job(task):
    from .interactions import plip_profile, quality
    path = Path(task["folder"]) / "EVALUATION.json"
    if path.exists():
        result = read(path)
        if result["binding"] != task["binding"]:
            raise ValueError("Terminal evaluation source binding differs")
        if not result.get("evaluation_error"):
            return result
        frozen(path.with_name("EVALUATION_FAILED_" + value_hash(result)[:16] + ".json"), result)
    condition = task["condition"]
    result = dict(binding=task["binding"], node_id=task["node_id"], seed=task["seed"], replicate=task["replicate"],
                  pic50=task.get("pic50"), build_success=bool(task.get("molblock")), fusion_success=False)
    if not task.get("molblock"):
        result["failure_reason"] = "native_build_failure"
        write(path, result)
        return result
    molecule = Chem.MolFromMolBlock(task["molblock"], removeHs=True)
    try:
        if molecule is None:
            raise ValueError("Saved native structure is not chemically valid")
        result.update(molecular_properties(molecule))
        result["connected"] = len(Chem.GetMolFrags(molecule)) == 1
        target_coordinates, target_elements = protein_atoms(task["protein"])
        result["protein_clash"] = clashes(molecule.GetConformer().GetPositions(),
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], target_coordinates, target_elements, task["clash_ratio"])
        fixed = Chem.MolFromMolBlock(condition["fixed_molblock"], removeHs=True)
        if molecule.GetNumAtoms() < fixed.GetNumAtoms():
            raise ValueError("Terminal has fewer atoms than the condition")
        chemistry_retained = True
        for atom_index in range(fixed.GetNumAtoms()):
            actual, expected = molecule.GetAtomWithIdx(atom_index), fixed.GetAtomWithIdx(atom_index)
            if (actual.GetAtomicNum(), actual.GetFormalCharge(), actual.GetIsAromatic()) != (
                    expected.GetAtomicNum(), expected.GetFormalCharge(), expected.GetIsAromatic()):
                chemistry_retained = False
        for begin in range(fixed.GetNumAtoms()):
            for end in range(begin + 1, fixed.GetNumAtoms()):
                actual = molecule.GetBondBetweenAtoms(begin, end)
                expected = fixed.GetBondBetweenAtoms(begin, end)
                if (str(actual.GetBondType()) if actual else None) != (str(expected.GetBondType()) if expected else None):
                    chemistry_retained = False
        result["condition_chemistry_retained"] = chemistry_retained
        parent_rmsd = {}
        for parent in condition["selected_contacts"]:
            indices = [position for position, source in enumerate(condition["sources"])
                       if source["parent_identifier"] == parent["parent_identifier"]]
            parent_rmsd[parent["parent_identifier"]] = frame_rmsd(molecule.GetConformer().GetPositions()[indices],
                                                                     fixed.GetConformer().GetPositions()[indices])
        result["condition_rmsd_protein_frame_A"] = parent_rmsd
        result["posebusters"] = quality(molecule, task["protein"])
        profile = plip_profile(molecule, task["protein"], Path(task["folder"]) / "plip")
        recovered = {}
        for parent in condition["selected_contacts"]:
            source_indices = {position for position, source in enumerate(condition["sources"])
                              if source["parent_identifier"] == parent["parent_identifier"]}
            expected = {tuple(fingerprint) for fingerprint in parent["fingerprints"]}
            contacts = [contact for contact in profile["interactions"]
                        if tuple(contact["fingerprint"]) in expected and set(contact["ligand_atoms"]) <= source_indices]
            recovered[parent["parent_identifier"]] = contacts
        result["recovered_interactions"] = recovered
        result["fusion_success"] = bool(result["connected"] and chemistry_retained
            and not result["protein_clash"]["severe"] and all(contacts for contacts in recovered.values())
            and all(score < task["rmsd_limit_A"] for score in parent_rmsd.values())
            and task.get("pic50") is not None and math.isfinite(task["pic50"]))
        if not result["fusion_success"]:
            result["failure_reason"] = "terminal_did_not_preserve_both_interaction_regions"
    except Exception as error:
        result["evaluation_error"] = dict(type=type(error).__name__, message=str(error)[:2000])
        result["failure_reason"] = "terminal_measurement_unavailable"
    write(path, result)
    return result


def table(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({name for row in rows for name in row})
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: str(value) if isinstance(value, (dict, list)) else value for name, value in row.items()})


def summarize(rows, parent_scores):
    successful = [row for row in rows if row.get("fusion_success")]
    scores = [row["pic50"] for row in successful]
    return dict(attempts=len(rows), build_success=sum(row.get("build_success", False) for row in rows),
                successful_fusions=len(successful), evaluable_pic50=sum(row.get("pic50") is not None for row in rows),
                evaluation_errors=sum(bool(row.get("evaluation_error")) for row in rows),
                valid_fusion_mean_pic50=float(numpy.mean(scores)) if scores else None,
                valid_fusion_best_pic50=max(scores) if scores else None,
                parent_pic50=parent_scores,
                mean_improvement_over_parents=[float(numpy.mean(scores)) - score for score in parent_scores] if scores else None,
                best_improvement_over_parents=[max(scores) - score for score in parent_scores] if scores else None)
