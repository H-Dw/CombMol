"""Independently audit a frozen evidence export; never sample or call a provider."""
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path

import numpy
from rdkit import Chem

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from combmol.geometry import clashes, topology
from combmol.io import value_hash
from combmol.provider import audit_fusions, audit_screen


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def table(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


class Evidence:
    def __init__(self, archive, manifest):
        self.manifest = json.loads(Path(manifest).read_text())
        assert checksum(archive) == self.manifest["archive_sha256"]
        with tarfile.open(archive, "r:gz") as stream:
            self.files = {item.name: stream.extractfile(item).read() for item in stream if item.isfile()}
        assert set(self.files) == set(self.manifest["files"])
        assert all(hashlib.sha256(payload).hexdigest() == self.manifest["files"][name] for name, payload in self.files.items())

    def read(self, name):
        return json.loads(self.files[name])

    def exists(self, name):
        return name in self.files


def selected_result(evidence, prefix):
    names = [name for name in evidence.files if name.startswith(prefix + "/") and name.endswith("/RESULT.json")]
    if evidence.exists(prefix + "/RESULT.json"):
        name = prefix + "/RESULT.json"
    else:
        assert len(names) == 1, (prefix, names)
        name = names[0]
    result = evidence.read(name)
    folder = name.rsplit("/", 1)[0]
    responses = [evidence.read(name) for name in evidence.files if name.startswith(folder + "/RESPONSE_")]
    complete = []
    for response in responses:
        if response.get("finish_reason") == "stop":
            try:
                complete.append(json.loads(response["content"]))
            except (ValueError, TypeError):
                pass
    assert result in complete, "A final decision must be a verbatim submitted JSON result"
    return result, evidence.read(folder + "/REQUEST.json")


def distribution(rows, name):
    values = [row[name] for row in rows if row.get(name) is not None]
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                standard_deviation=statistics.pstdev(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def storage_audit(evidence, root, output):
    """Optional server-side checks of original archives and tensors left on the server."""
    import torch
    from combmol.native import tensor_hash, tensor_leaves
    root = Path(root)
    config = evidence.read("CONFIG.json")
    launch = evidence.read("LAUNCH.json")
    repository = Path(launch["repository"])
    upstream = json.loads((repository / "third_party/UPSTREAM.json").read_text())
    assert upstream["commit"] == config["upstream_commit"]
    assert all(checksum(repository / "third_party/flowr_root" / relative) == expected for relative, expected in upstream["files"].items())
    assert checksum(config["checkpoint"]) == config["checkpoint_sha256"]
    import subprocess
    assert subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip() == launch["revision"]
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert boot_id == launch["boot_id"]
    process = Path(f"/proc/{launch['pid']}/stat")
    if process.exists():
        fields = process.read_text().rsplit(")", 1)[1].split()
        assert fields[0] == "Z" or fields[19] != str(launch["start_ticks"]), "Generation process is still alive"
    batch_count = 0
    single_atom_coordinate_degeneracy = 0
    coordinate_error = 0.
    for target in config["targets"]:
        prefix = "targets/" + target
        index = evidence.read(prefix + "/sources/INDEX.json")
        assert checksum(Path(config["dataset_root"]) / index["archive_relative"]) == index["archive_sha256"]
        assert all(checksum(root / prefix / "sources" / relative) == expected for relative, expected in index["member_hashes"].items())
        for condition in evidence.read(prefix + "/selection/CONDITIONS.json"):
            group = prefix + "/fusions/" + condition["proposal"]["identifier"]
            ledger = evidence.read(group + "/ATTEMPTS.json")
            for seed in config["seeds"]:
                batch = group + f"/seed_{seed}"
                initial = torch.load(root / batch / "INITIALIZATION.pt", map_location="cpu", weights_only=True)
                identity = evidence.read(batch + "/INITIALIZATION.json")
                assert tensor_hash(initial["prior"]) == identity["hash"]
                free = initial["prior"]["mask"].bool() & ~initial["prior"]["fragment_mask"].bool()
                coordinate_states = {initial["prior"]["coords"][row][free[row]].contiguous().numpy().tobytes() for row in range(5)}
                full_states = set()
                for row in range(5):
                    state = hashlib.sha256()
                    for labels, value in tensor_leaves(initial["prior"]):
                        if value.ndim and value.shape[0] == 5:
                            state.update(str(labels).encode())
                            state.update(value[row].contiguous().numpy().tobytes())
                    full_states.add(state.hexdigest())
                assert len(full_states) == 5, "A full initialized input was duplicated within a batch"
                if len(coordinate_states) < 5:
                    # Native zero-COM Gaussian coordinates have no variability with one free atom.
                    assert bool((free.sum(-1) == 1).all())
                    single_atom_coordinate_degeneracy += 1
                terminal_path = root / batch / "TERMINALS.pt"
                terminal_hash = checksum(terminal_path)
                terminal = torch.load(terminal_path, map_location="cpu", weights_only=True)
                assert terminal["coords"].shape[0] == 5
                for row in evidence.read(batch + "/TERMINAL_RECORDS.json"):
                    slot = ledger["slots"][f"{seed}:{row['replicate']}"]
                    assert Path(slot["terminal_file"]).resolve() == terminal_path.resolve()
                    assert slot["terminal_sha256"] == terminal_hash
                    assert slot["initialization_hash"] == identity["hash"]
                    if row["molblock"]:
                        molecule = Chem.MolFromMolBlock(row["molblock"], removeHs=True)
                        coordinates = terminal["coords"][row["replicate"], :molecule.GetNumAtoms()].numpy()
                        delta = float(numpy.abs(coordinates - molecule.GetConformer().GetPositions()).max())
                        coordinate_error = max(coordinate_error, delta)
                        assert delta <= 0.000051, "Stored molecule differs from the returned terminal coordinates"
                batch_count += 1
    provenance = evidence.read("recoveries/tensor_and_screen_001/RECOVERY.json")
    assert all(checksum(root / relative) == expected for relative, expected in provenance["frozen_selection_sha256"].items())
    result = dict(passed=True, upstream_files=len(upstream["files"]), targets=2, terminal_batches=batch_count,
                  all_batches_have_five_distinct_full_inputs=True,
                  single_free_atom_coordinate_degeneracy_batches=single_atom_coordinate_degeneracy,
                  frozen_prior_selections=len(provenance["frozen_selection_sha256"]),
                  max_molblock_rounding_error_A=coordinate_error, checkpoint_verified=True,
                  scientific_revision=launch["revision"], process_ended=True)
    save(output / "storage_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--storage-root")
    args = parser.parse_args()
    evidence = Evidence(args.archive, args.manifest)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    config, final = evidence.read("CONFIG.json"), evidence.read("runs/attempt_003/FINISHED.json")
    assert final["all_success"] and final["no_gradient_guidance"] and not final["final_inpaint"]
    assert not config["apply_guidance"] and not config["final_inpaint"]
    assert config["seeds"] == [42, 100045, 200048, 300051, 400054]
    assert config["replicates"] == 5 and config["integration_steps"] == 100
    assert config["condition_rmsd_limit_A"] == .5 and config["clash_ratio"] == .75
    assert value_hash(config) == evidence.read("PLAN.json")["configuration_hash"]
    assert final["revision"] == evidence.read("LAUNCH.json")["revision"]
    all_rows, parents_table, groups_table, target_table = [], [], [], []
    maximum_rmsd_difference = 0.
    for target in config["targets"]:
        prefix = "targets/" + target
        index = evidence.read(prefix + "/sources/INDEX.json")
        source_map = {item["identifier"]: item for item in index["candidates"]}
        assert len(index["records"]) == len({item["node_id"] for item in index["records"]}) == 100
        scores = evidence.read(prefix + "/selection/SCORES.json")
        assert len(scores["scores"]) == len(index["candidates"]) and all(math.isfinite(value) for value in scores["scores"])
        assert scores["sources"] == [item["evidence_id"] for item in index["candidates"]]
        source_scores = {item["identifier"]: score for item, score in zip(index["candidates"], scores["scores"])}
        parent_receipt = evidence.read(prefix + "/selection/PARENTS.json")
        parents = parent_receipt["parents"]
        assert len(parents) <= 8 and len({topology(Chem.MolFromMolBlock(item["molblock"], removeHs=True)) for item in parents}) == len(parents)
        parent_map = {item["identifier"]: item for item in parents}
        for position, parent in enumerate(parents, 1):
            assert parent["evidence_id"] == source_map[parent["identifier"]]["evidence_id"]
            assert parent["molblock"] == source_map[parent["identifier"]]["molblock"]
            assert parent["pic50"] == source_scores[parent["identifier"]]
            parents_table.append(dict(target=target, parent=f"parent-{position:02d}", identifier=parent["identifier"],
                pic50=parent["pic50"], SAScore=parent["basic_quality"]["SAScore"], QEDScore=parent["basic_quality"]["QEDScore"],
                heavy_atoms=parent["basic_quality"]["heavy_atoms"], posebusters_pass=parent["diagnostics"].get("posebusters_pass")))
        screen_prefixes = sorted({name.rsplit("/", 1)[0] for name in evidence.files if name.startswith(prefix + "/selection/llm_screen/") and name.endswith("/REQUEST.json") and "/recoveries/" not in name})
        accepted = set()
        for screen in screen_prefixes:
            result, request = selected_result(evidence, screen)
            audit_screen(result, request["packet"]["candidates"])
            accepted.update(row["candidate_identifier"] for row in result["decisions"] if row["keep"])
        ranked, graphs = [], set()
        for identifier in sorted(accepted, key=lambda identifier: (-source_scores[identifier], identifier)):
            graph = topology(Chem.MolFromMolBlock(source_map[identifier]["molblock"], removeHs=True))
            if graph not in graphs:
                graphs.add(graph)
                ranked.append(identifier)
        assert [item["identifier"] for item in parents] == ranked[:8]
        conditions = evidence.read(prefix + "/selection/CONDITIONS.json")
        fusion_result, fusion_request = selected_result(evidence, prefix + "/selection/llm_fusion")
        audit_config = dict(config, repository=str(REPOSITORY if (REPOSITORY / "third_party").exists() else Path(evidence.read("LAUNCH.json")["repository"])))
        audit_fusions(fusion_result, parents, evidence.read(prefix + "/selection/REGIONS.json"), audit_config)
        settings = evidence.read(prefix + "/native/NATIVE_SETTINGS.json")
        assert settings["model_inpainting_mode"] and settings["model_inpainting_mode_inf"] == "fragment"
        assert all(not settings["arguments"][name] for name in ["final_inpaint", "rotation_alignment", "permutation_alignment", "use_sde_simulation"])
        assert [item["proposal"] for item in conditions] == fusion_result["proposals"]
        assert len(conditions) <= 16
        pairs = set()
        protein_payload = evidence.files[prefix + "/sources/inputs/target_protein.pdb"]
        protein_sha = hashlib.sha256(protein_payload).hexdigest()
        protein_coordinates, protein_elements = [], []
        for line in protein_payload.decode().splitlines():
            if line.startswith(("ATOM  ", "HETATM")):
                symbol = line[76:78].strip().capitalize()
                if not symbol:
                    symbol = line[12:16].strip().lstrip("0123456789")[0].upper()
                if symbol not in ("H", "D"):
                    protein_coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    protein_elements.append(Chem.GetPeriodicTable().GetAtomicNumber(symbol))
        target_rows = []
        for condition in conditions:
            assert value_hash({key: value for key, value in condition.items() if key != "condition_hash"}) == condition["condition_hash"]
            assert not condition["apply_guidance"] and not condition["final_inpaint"]
            group_id = condition["proposal"]["identifier"]
            pair = tuple(sorted(item["parent_identifier"] for item in condition["proposal"]["parent_regions"]))
            assert len(set(pair)) == 2 and pair not in pairs
            pairs.add(pair)
            fixed = Chem.MolFromMolBlock(condition["fixed_molblock"], removeHs=True)
            fixed_coordinates = fixed.GetConformer().GetPositions()
            for position, source in enumerate(condition["sources"]):
                source_molecule = Chem.MolFromMolBlock(parent_map[source["parent_identifier"]]["molblock"], removeHs=True)
                assert numpy.allclose(fixed_coordinates[position], source_molecule.GetConformer().GetPositions()[source["source_atom"]], atol=1e-8)
            group = prefix + "/fusions/" + group_id
            ledger = evidence.read(group + "/ATTEMPTS.json")
            assert ledger["binding"] == dict(condition_hash=condition["condition_hash"], seeds=config["seeds"], replicates=5)
            assert set(ledger["slots"]) == {f"{seed}:{replicate}" for seed in config["seeds"] for replicate in range(5)}
            assert all(slot["state"] == "complete" and slot["revision"] == final["revision"] for slot in ledger["slots"].values())
            group_rows = []
            for seed in config["seeds"]:
                batch = group + f"/seed_{seed}"
                rows = evidence.read(batch + "/TERMINAL_RECORDS.json")
                assert {(row["seed"], row["replicate"]) for row in rows} == {(seed, replicate) for replicate in range(5)} and len(rows) == 5
                endpoints = evidence.read(batch + "/ENDPOINT_SCORES.json")
                for terminal in rows:
                    node = terminal["node_id"]
                    measured = evidence.read(batch + "/" + node + "/EVALUATION.json")
                    score = endpoints.get(node)
                    assert measured["binding"] == value_hash(dict(condition=condition["condition_hash"], molblock=terminal["molblock"], score=score, protein=protein_sha, protocol=config["protocol"]))
                    assert measured["pic50"] == score and not measured.get("evaluation_error")
                    built = bool(terminal["molblock"])
                    assert measured["build_success"] == built
                    result = dict(target=target, fusion=group_id, node_id=node, seed=seed, replicate=terminal["replicate"],
                                  build_success=built, fusion_success=measured["fusion_success"], pic50=score,
                                  evaluation_path=batch + "/" + node + "/EVALUATION.json")
                    if built:
                        molecule = Chem.MolFromMolBlock(terminal["molblock"], removeHs=True)
                        assert molecule is not None and score is not None and math.isfinite(score)
                        coordinates = molecule.GetConformer().GetPositions()
                        connected = len(Chem.GetMolFrags(molecule)) == 1
                        chemistry = molecule.GetNumAtoms() >= fixed.GetNumAtoms()
                        for position in range(fixed.GetNumAtoms()):
                            actual, expected = molecule.GetAtomWithIdx(position), fixed.GetAtomWithIdx(position)
                            chemistry &= (actual.GetAtomicNum(), actual.GetFormalCharge(), actual.GetIsAromatic()) == (expected.GetAtomicNum(), expected.GetFormalCharge(), expected.GetIsAromatic())
                        for begin in range(fixed.GetNumAtoms()):
                            for end in range(begin + 1, fixed.GetNumAtoms()):
                                actual, expected = molecule.GetBondBetweenAtoms(begin, end), fixed.GetBondBetweenAtoms(begin, end)
                                chemistry &= (str(actual.GetBondType()) if actual else None) == (str(expected.GetBondType()) if expected else None)
                        rmsds, contacts = {}, {}
                        profile = evidence.read(batch + "/" + node + "/plip/interactions.json")
                        for parent in condition["selected_contacts"]:
                            indices = [position for position, source in enumerate(condition["sources"]) if source["parent_identifier"] == parent["parent_identifier"]]
                            rmsd = float(numpy.sqrt(numpy.mean(numpy.sum((coordinates[indices] - fixed_coordinates[indices]) ** 2, axis=-1))))
                            rmsds[parent["parent_identifier"]] = rmsd
                            maximum_rmsd_difference = max(maximum_rmsd_difference, abs(rmsd - measured["condition_rmsd_protein_frame_A"][parent["parent_identifier"]]))
                            expected = {tuple(item) for item in parent["fingerprints"]}
                            contacts[parent["parent_identifier"]] = [item for item in profile["interactions"] if tuple(item["fingerprint"]) in expected and set(item["ligand_atoms"]) <= set(indices)]
                        clash = clashes(coordinates, [atom.GetAtomicNum() for atom in molecule.GetAtoms()], protein_coordinates, protein_elements, config["clash_ratio"])
                        assert connected == measured["connected"] and bool(chemistry) == measured["condition_chemistry_retained"]
                        assert clash["severe"] == measured["protein_clash"]["severe"] and contacts == measured["recovered_interactions"]
                        qualified = connected and chemistry and not clash["severe"] and all(value < 0.5 for value in rmsds.values()) and all(contacts.values())
                        assert bool(qualified) == measured["fusion_success"]
                        result.update(SAScore=measured["SAScore"], QEDScore=measured["QEDScore"], heavy_atoms=molecule.GetNumAtoms(),
                            connected=connected, chemistry_retained=bool(chemistry), max_condition_rmsd_A=max(rmsds.values()),
                            protein_clash=clash["severe"], both_contacts_recovered=all(contacts.values()),
                            posebusters_evaluable=measured["posebusters"].get("posebusters_evaluable"), posebusters_pass=measured["posebusters"].get("posebusters_pass"),
                            topology=topology(molecule), exceeds_both_parents=score > max(parent_map[name]["pic50"] for name in pair))
                    else:
                        assert score is None and not measured["fusion_success"]
                    group_rows.append(result)
            assert len(group_rows) == 25
            qualified = [row for row in group_rows if row["fusion_success"]]
            original = evidence.read(group + "/SUMMARY.json")
            assert original["attempts"] == 25 and original["successful_fusions"] == len(qualified)
            assert original["build_success"] == sum(row["build_success"] for row in group_rows)
            mean = statistics.mean(row["pic50"] for row in qualified) if qualified else None
            best = max(qualified, key=lambda row: row["pic50"]) if qualified else None
            assert mean is None or abs(mean - original["valid_fusion_mean_pic50"]) < 1e-12
            assert (best["pic50"] if best else None) == original["valid_fusion_best_pic50"]
            groups_table.append(dict(target=target, fusion=group_id, attempts=25, built=original["build_success"], qualified=len(qualified),
                qualified_rate=len(qualified) / 25, mean_pic50=mean, best_pic50=best["pic50"] if best else None,
                parent_left=pair[0], parent_right=pair[1], parent_left_pic50=parent_map[pair[0]]["pic50"], parent_right_pic50=parent_map[pair[1]]["pic50"],
                best_delta_over_better_parent=best["pic50"] - max(parent_map[name]["pic50"] for name in pair) if best else None,
                qualified_above_both_parents=sum(row.get("exceeds_both_parents", False) for row in qualified)))
            target_rows.extend(group_rows)
        built = [row for row in target_rows if row["build_success"]]
        qualified = [row for row in target_rows if row["fusion_success"]]
        ranked_qualified = sorted(qualified, key=lambda row: (-row["pic50"], row["node_id"]))
        best = ranked_qualified[0] if qualified else None
        top = ranked_qualified[:max(1, math.ceil(len(qualified) * .1))]
        best_parent = max(parent["pic50"] for parent in parents)
        row = dict(target=target, parents=len(parents), accepted_parent_poses=len(accepted), conditions=len(conditions), attempts=len(target_rows),
            built=len(built), build_rate=len(built) / len(target_rows), qualified=len(qualified), qualified_rate=len(qualified) / len(target_rows),
            qualified_pic50_mean=statistics.mean(item["pic50"] for item in qualified) if qualified else None,
            qualified_pic50_top10_mean=statistics.mean(item["pic50"] for item in top) if top else None, qualified_top10_count=len(top),
            qualified_pic50_best=best["pic50"] if best else None, best_qualified_node=best["node_id"] if best else None,
            best_qualified_fusion=best["fusion"] if best else None, best_qualified_SAScore=best.get("SAScore") if best else None,
            best_qualified_heavy_atoms=best.get("heavy_atoms") if best else None, best_qualified_posebusters_pass=best.get("posebusters_pass") if best else None,
            best_parent_pic50=best_parent, best_delta_over_best_parent=best["pic50"] - best_parent if best else None,
            qualified_exceeding_both_parents=sum(item.get("exceeds_both_parents", False) for item in qualified),
            qualified_posebusters_pass=sum(item.get("posebusters_pass") is True for item in qualified),
            qualified_unique_topologies=len({item["topology"] for item in qualified}))
        for name in ["pic50", "SAScore", "QEDScore", "heavy_atoms"]:
            for label, population in [("built", built), ("qualified", qualified), ("qualified_top10_by_pic50", top)]:
                measured = distribution(population, name)
                for statistic in ["mean", "median", "standard_deviation"]:
                    row[label + "_" + name + "_" + statistic] = measured[statistic]
        target_table.append(row)
        all_rows.extend(target_rows)
    assert len(all_rows) == final["spent_attempts"] <= 800
    assert len({(row["target"], row["fusion"], row["seed"], row["replicate"]) for row in all_rows}) == len(all_rows)
    assert maximum_rmsd_difference < 1e-12
    table(output / "candidate_metrics.csv", all_rows)
    table(output / "parent_metrics.csv", parents_table)
    table(output / "fusion_summary.csv", groups_table)
    table(output / "target_summary.csv", target_table)
    audit = dict(passed=True, scientific_revision=final["revision"], attempts=len(all_rows), conditions=len(groups_table), targets=target_table,
        evidence_files_verified=len(evidence.files), maximum_recomputed_condition_rmsd_difference_A=maximum_rmsd_difference,
        unique_slots=True, no_missing_terminal_evaluations=True, unchanged_scientific_protocol=True,
        mean_policy="candidate-weighted arithmetic mean within explicitly labelled cohorts")
    if args.storage_root:
        audit["storage"] = storage_audit(evidence, args.storage_root, output)
    save(output / "analysis.json", audit)
    print(json.dumps(dict(passed=True, attempts=len(all_rows), targets=target_table), ensure_ascii=False))


if __name__ == "__main__":
    main()
