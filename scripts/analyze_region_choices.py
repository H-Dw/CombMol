"""Measure saved CombMol region geometry, interaction coverage and attachment recovery."""
import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy
from rdkit import Chem, rdBase
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from analyze_campaign import Evidence, table
from combmol.geometry import proper_alignment, topology
from combmol.io import value_hash


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def fitted_rmsd(moving, reference):
    rotation, translation = proper_alignment(moving, reference)
    return float(numpy.sqrt(numpy.mean(numpy.sum((moving @ rotation + translation - reference) ** 2, axis=-1))))


def feature_key(contact):
    return tuple(contact["fingerprint"])


def contact_key(contact, mapping=None):
    atoms = contact["ligand_atoms"]
    if mapping is not None:
        atoms = [mapping[atom] for atom in atoms]
    return feature_key(contact), tuple(sorted(atoms))


def jaccard(left, right):
    return len(left & right) / len(left | right) if left | right else None


def median(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return statistics.median(values) if values else None


def bridging_path(molecule, left_ports, right_ports, fixed_count):
    for start in left_ports:
        queue = collections.deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            for atom in molecule.GetAtomWithIdx(path[-1]).GetNeighbors():
                neighbor = atom.GetIdx()
                if neighbor in right_ports and len(path) > 1:
                    return path + [neighbor]
                if neighbor >= fixed_count and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()
    evidence = Evidence(options.evidence, options.manifest)
    output = Path(options.out)
    output.mkdir(parents=True, exist_ok=True)
    group_rows, terminal_rows, details = [], [], []
    config = evidence.read("CONFIG.json")
    for target in config["targets"]:
        prefix = "targets/" + target
        parents = evidence.read(prefix + "/selection/PARENTS.json")["parents"]
        parent_map = {item["identifier"]: item for item in parents}
        parent_alias = {item["identifier"]: f"parent-{position:02d}" for position, item in enumerate(parents, 1)}
        molecules = {key: Chem.MolFromMolBlock(item["molblock"], removeHs=True) for key, item in parent_map.items()}
        protein = []
        for line in evidence.files[prefix + "/sources/inputs/target_protein.pdb"].decode().splitlines():
            if line.startswith(("ATOM  ", "HETATM")) and line[76:78].strip().upper() not in ("H", "D"):
                protein.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        tree = cKDTree(protein)
        for condition in evidence.read(prefix + "/selection/CONDITIONS.json"):
            identifier = condition["proposal"]["identifier"]
            group = prefix + "/fusions/" + identifier
            fixed = Chem.MolFromMolBlock(condition["fixed_molblock"], removeHs=True)
            fixed_count = fixed.GetNumAtoms()
            positions = fixed.GetConformer().GetPositions()
            parent_ids = [item["parent_identifier"] for item in condition["selected_contacts"]]
            indices = [[position for position, source in enumerate(condition["sources"]) if source["parent_identifier"] == parent] for parent in parent_ids]
            mappings = [{source["source_atom"]: position for position, source in enumerate(condition["sources"]) if source["parent_identifier"] == parent} for parent in parent_ids]
            distances = numpy.linalg.norm(positions[indices[0]][:, None] - positions[indices[1]][None], axis=-1)
            original_centroid_distance = float(numpy.linalg.norm(positions[indices[0]].mean(0) - positions[indices[1]].mean(0)))
            ports, selected_contacts = [], []
            full_features, selected_features = [], []
            for side, parent in enumerate(parent_ids):
                molecule = molecules[parent]
                coordinates = molecule.GetConformer().GetPositions()
                source_atoms = set(mappings[side])
                boundary = []
                for bond in molecule.GetBonds():
                    begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if (begin in source_atoms) != (end in source_atoms):
                        inside, outside = (begin, end) if begin in source_atoms else (end, begin)
                        vector = coordinates[outside] - coordinates[inside]
                        boundary.append(dict(source_atom=inside, omitted_atom=outside, fixed_slot=mappings[side][inside],
                            source_bond_type=str(bond.GetBondType()), position=coordinates[inside].tolist(), exit_vector=vector.tolist()))
                ports.append(boundary)
                chosen = set(condition["selected_contacts"][side]["evidence_ids"])
                selected_contacts.append([row for row in parent_map[parent]["profile"]["interactions"] if row["evidence_id"] in chosen])
                full_features.append({feature_key(row) for row in parent_map[parent]["profile"]["interactions"]})
                selected_features.append({feature_key(row) for row in selected_contacts[-1]})
            port_pairs = []
            for left in ports[0]:
                for right in ports[1]:
                    difference = numpy.array(right["position"]) - left["position"]
                    distance = float(numpy.linalg.norm(difference))
                    direction = difference / distance
                    cos_left = float(numpy.dot(left["exit_vector"], direction) / numpy.linalg.norm(left["exit_vector"]))
                    cos_right = float(numpy.dot(right["exit_vector"], -direction) / numpy.linalg.norm(right["exit_vector"]))
                    port_pairs.append(dict(left_source_atom=left["source_atom"], right_source_atom=right["source_atom"], distance_A=distance,
                        left_angle_deg=math.degrees(math.acos(numpy.clip(cos_left, -1, 1))),
                        right_angle_deg=math.degrees(math.acos(numpy.clip(cos_right, -1, 1))), min_cosine=min(cos_left, cos_right)))
            nearest = min(port_pairs, key=lambda row: row["distance_A"])
            best_facing = max(port_pairs, key=lambda row: row["min_cosine"])
            group_row = dict(target=target, fusion=identifier, left_parent=parent_ids[0], right_parent=parent_ids[1],
                left_alias=parent_alias[parent_ids[0]], right_alias=parent_alias[parent_ids[1]], fixed_atoms=fixed_count,
                fixed_components=len(Chem.GetMolFrags(fixed)), selected_left_atoms=len(indices[0]), selected_right_atoms=len(indices[1]),
                min_inter_region_distance_A=float(distances.min()), min_vdw_ratio=condition["mutual_clash"]["min_ratio"],
                centroid_distance_A=original_centroid_distance, nearest_port_distance_A=nearest["distance_A"],
                nearest_left_exit_angle_deg=nearest["left_angle_deg"], nearest_right_exit_angle_deg=nearest["right_angle_deg"],
                best_pair_min_cosine=best_facing["min_cosine"], best_facing_distance_A=best_facing["distance_A"],
                any_pair_both_forward=any(row["left_angle_deg"] < 90 and row["right_angle_deg"] < 90 for row in port_pairs),
                left_ports=len(ports[0]), right_ports=len(ports[1]),
                selected_fingerprint_overlap=jaccard(*selected_features),
                selected_residue_overlap=jaccard(*[{tuple(item[1:4]) for item in features} for features in selected_features]),
                left_full_features=len(full_features[0]), right_full_features=len(full_features[1]),
                left_selected_features=len(selected_features[0]), right_selected_features=len(selected_features[1]),
                left_feature_coverage=len(selected_features[0]) / len(full_features[0]),
                right_feature_coverage=len(selected_features[1]) / len(full_features[1]),
                minimum_unique_feature_contribution=min(len(selected_features[0] - selected_features[1]), len(selected_features[1] - selected_features[0])),
                selected_topology=topology(fixed), parent_mean_pic50=statistics.mean(parent_map[parent]["pic50"] for parent in parent_ids),
                parent_better_pic50=max(parent_map[parent]["pic50"] for parent in parent_ids))
            omitted_shared = (full_features[0] & full_features[1]) - (selected_features[0] | selected_features[1])
            group_row["omitted_shared_feature_count"] = len(omitted_shared)
            group_row["omitted_shared_features"] = sorted(omitted_shared)
            group_row["conditions_GLY152_acceptor"] = ("hbond", "A", "152", "GLY", "ligand_acceptor") in selected_features[0] | selected_features[1]
            nearest_left = next(port for port in ports[0] if port["source_atom"] == nearest["left_source_atom"])
            nearest_right = next(port for port in ports[1] if port["source_atom"] == nearest["right_source_atom"])
            segment = numpy.linspace(nearest_left["position"], nearest_right["position"], max(4, math.ceil(nearest["distance_A"] / .25)))[1:-1]
            group_row["straight_line_min_protein_distance_A"] = float(tree.query(segment)[0].min())
            nodes = []
            for seed in config["seeds"]:
                batch = group + f"/seed_{seed}"
                initialization = evidence.read(batch + "/INITIALIZATION.json")
                for row in evidence.read(batch + "/TERMINAL_RECORDS.json"):
                    result = dict(target=target, fusion=identifier, node_id=row["node_id"], seed=seed, replicate=row["replicate"],
                        build_success=bool(row["molblock"]), free_atoms=initialization["atom_count"] - fixed_count)
                    if row["molblock"]:
                        measured = evidence.read(batch + "/" + row["node_id"] + "/EVALUATION.json")
                        molecule = Chem.MolFromMolBlock(row["molblock"], removeHs=True)
                        coordinates = molecule.GetConformer().GetPositions()
                        assert molecule.GetNumAtoms() >= fixed_count
                        fixed_positions = coordinates[:fixed_count]
                        rotation, translation = proper_alignment(fixed_positions, positions)
                        jointly_fitted = fixed_positions @ rotation + translation
                        joint_rmsd = float(numpy.sqrt(numpy.mean(numpy.sum((jointly_fitted - positions) ** 2, axis=-1))))
                        joint_side_max = max(float(numpy.sqrt(numpy.mean(numpy.sum((jointly_fitted[side] - positions[side]) ** 2, axis=-1)))) for side in indices)
                        parent_rmsds = [fitted_rmsd(fixed_positions[side], positions[side]) for side in indices]
                        cross = numpy.linalg.norm(fixed_positions[indices[0]][:, None] - fixed_positions[indices[1]][None], axis=-1)
                        centroid = float(numpy.linalg.norm(fixed_positions[indices[0]].mean(0) - fixed_positions[indices[1]].mean(0)))
                        result.update(pic50=measured["pic50"], pic50_delta_parent_mean=measured["pic50"] - group_row["parent_mean_pic50"],
                            pic50_delta_better_parent=measured["pic50"] - group_row["parent_better_pic50"],
                            chemistry_retained=measured["condition_chemistry_retained"], connected=measured["connected"],
                            fusion_success=measured["fusion_success"], posebusters_pass=measured["posebusters"].get("posebusters_pass"),
                            frame_max_rmsd_A=max(measured["condition_rmsd_protein_frame_A"].values()),
                            joint_fit_rmsd_A=joint_rmsd, joint_fit_max_side_rmsd_A=joint_side_max, parent_fit_max_rmsd_A=max(parent_rmsds),
                            cross_distance_rms_error_A=float(numpy.sqrt(numpy.mean((cross - distances) ** 2))),
                            centroid_separation_change_A=abs(centroid - original_centroid_distance),
                            condition_centroid_displacement_A=float(numpy.linalg.norm(fixed_positions.mean(0) - positions.mean(0))))
                        port_slots = [set(port["fixed_slot"] for port in boundary) for boundary in ports]
                        allowed_slots = set.union(*port_slots)
                        side_lookup = {atom: side for side, atoms in enumerate(indices) for atom in atoms}
                        cross_fixed, external_edges, off_port_edges = 0, 0, 0
                        for bond in molecule.GetBonds():
                            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            if begin < fixed_count and end < fixed_count and side_lookup[begin] != side_lookup[end]:
                                cross_fixed += 1
                            if (begin < fixed_count) != (end < fixed_count):
                                fixed_atom = begin if begin < fixed_count else end
                                external_edges += 1
                                off_port_edges += fixed_atom not in allowed_slots
                        path = bridging_path(molecule, port_slots[0], port_slots[1], fixed_count)
                        result.update(cross_parent_fixed_bonds=cross_fixed, fixed_free_edges=external_edges,
                            off_port_edges=off_port_edges, port_bridge_exists=path is not None,
                            bridge_free_atoms=len(path) - 2 if path else None)
                        profile = evidence.read(batch + "/" + row["node_id"] + "/plip/interactions.json")
                        whole_features = {feature_key(contact) for contact in profile["interactions"]}
                        result["whole_GLY152_acceptor"] = ("hbond", "A", "152", "GLY", "ligand_acceptor") in whole_features
                        result["omitted_shared_features_recovered_fraction"] = len(whole_features & omitted_shared) / len(omitted_shared) if omitted_shared else None
                        feature_recovery, atom_recovery = [], []
                        selected_hbonds, recovered_hbonds = 0, 0
                        type_expected, type_recovered = collections.Counter(), collections.Counter()
                        for side in range(2):
                            actual = [contact for contact in profile["interactions"] if set(contact["ligand_atoms"]) <= set(indices[side])]
                            actual_features = {feature_key(contact) for contact in actual}
                            type_expected.update(feature[0] for feature in selected_features[side])
                            type_recovered.update(feature[0] for feature in actual_features & selected_features[side])
                            expected_atoms = {contact_key(contact, mappings[side]) for contact in selected_contacts[side]}
                            actual_atoms = {contact_key(contact) for contact in actual}
                            feature_recovery.append(len(actual_features & selected_features[side]) / len(selected_features[side]))
                            atom_recovery.append(len(actual_atoms & expected_atoms) / len(expected_atoms))
                            expected_hbonds = {feature for feature in selected_features[side] if feature[0] == "hbond"}
                            selected_hbonds += len(expected_hbonds)
                            recovered_hbonds += len(expected_hbonds & actual_features)
                        result.update(selected_feature_recovery_min=min(feature_recovery),
                            selected_atom_contact_recovery_min=min(atom_recovery),
                            all_selected_features_recovered=all(value == 1 for value in feature_recovery),
                            all_selected_atom_contacts_recovered=all(value == 1 for value in atom_recovery),
                            selected_hbond_features=selected_hbonds, recovered_hbond_features=recovered_hbonds,
                            all_selected_hbond_features_recovered=recovered_hbonds == selected_hbonds if selected_hbonds else None,
                            feature_types_expected=dict(type_expected), feature_types_recovered=dict(type_recovered))
                    nodes.append(result)
            built = [row for row in nodes if row["build_success"]]
            retained = [row for row in built if row["chemistry_retained"]]
            group_row.update(built=len(built), qualified=sum(row["fusion_success"] for row in built),
                qualified_rate=sum(row["fusion_success"] for row in built) / 25,
                mean_pic50=statistics.mean(row["pic50"] for row in built),
                mean_pic50_delta=statistics.mean(row["pic50_delta_parent_mean"] for row in built),
                mean_free_atoms=statistics.mean(row["free_atoms"] for row in nodes),
                median_joint_fit_rmsd_A=median(retained, "joint_fit_rmsd_A"),
                retained_chemistry_count=len(retained), off_port_candidates=sum(row["off_port_edges"] > 0 for row in retained),
                no_port_bridge_candidates=sum(not row["port_bridge_exists"] for row in retained))
            group_rows.append(group_row)
            terminal_rows.extend(nodes)
            details.append(dict(target=target, fusion=identifier, parent_ids=parent_ids,
                selected_source_atoms=[sorted(mapping) for mapping in mappings], ports=ports,
                actual_subgraphs=[dict(parent_id=parent,
                    fragment_smiles=Chem.MolFragmentToSmiles(molecules[parent], atomsToUse=sorted(mappings[side])),
                    element_counts=dict(collections.Counter(molecules[parent].GetAtomWithIdx(atom).GetSymbol() for atom in mappings[side])),
                    retained_source_feature_ids=[contact["evidence_id"] for contact in selected_contacts[side]],
                    omitted_full_parent_fingerprints=sorted(full_features[side] - selected_features[side]))
                    for side, parent in enumerate(parent_ids)],
                selected_contacts=selected_contacts, geometry=group_row, proposal=condition["proposal"],
                condition_hash=condition["condition_hash"]))
    summary = {}
    for target in config["targets"]:
        groups = [row for row in group_rows if row["target"] == target]
        nodes = [row for row in terminal_rows if row["target"] == target]
        built = [row for row in nodes if row["build_success"]]
        retained = [row for row in built if row["chemistry_retained"]]
        frame_fail = [row for row in retained if row["frame_max_rmsd_A"] >= .5]
        correlations = {}
        for field in ["min_inter_region_distance_A", "nearest_port_distance_A", "best_pair_min_cosine", "selected_fingerprint_overlap", "mean_free_atoms"]:
            correlations[field] = {}
            for outcome in ["qualified_rate", "mean_pic50_delta"]:
                coefficient = float(spearmanr([row[field] for row in groups], [row[outcome] for row in groups]).statistic) if len({row[field] for row in groups}) > 1 else float("nan")
                correlations[field][outcome] = coefficient if math.isfinite(coefficient) else None
        strata = {}
        type_summary = {}
        for label, population in [("chemistry_retained", retained), ("original_qualified", [row for row in built if row["fusion_success"]])]:
            expected, recovered = collections.Counter(), collections.Counter()
            for row in population:
                expected.update(row["feature_types_expected"])
                recovered.update(row["feature_types_recovered"])
            type_summary[label] = {name: dict(expected=count, recovered=recovered[name], fraction=recovered[name] / count) for name, count in expected.items()}
        for name, rows in [("all_built", built), ("chemistry_retained", retained),
                ("joint_fit_below_0_5", [row for row in retained if row["joint_fit_rmsd_A"] < .5]),
                ("all_selected_features", [row for row in retained if row["all_selected_features_recovered"]]),
                ("original_qualified", [row for row in built if row["fusion_success"]]),
                ("original_qualified_PB_pass", [row for row in built if row["fusion_success"] and row["posebusters_pass"]])]:
            strata[name] = dict(count=len(rows), mean_pic50=statistics.mean(row["pic50"] for row in rows) if rows else None,
                mean_paired_parent_delta=statistics.mean(row["pic50_delta_parent_mean"] for row in rows) if rows else None)
        summary[target] = dict(conditions=len(groups), built=len(built), chemistry_retained=len(retained),
            min_region_distance_range_A=[min(row["min_inter_region_distance_A"] for row in groups), max(row["min_inter_region_distance_A"] for row in groups)],
            min_vdw_ratio_range=[min(row["min_vdw_ratio"] for row in groups), max(row["min_vdw_ratio"] for row in groups)],
            port_distance_range_A=[min(row["nearest_port_distance_A"] for row in groups), max(row["nearest_port_distance_A"] for row in groups)],
            no_pair_both_forward=sum(not row["any_pair_both_forward"] for row in groups),
            unique_selected_topologies=len({row["selected_topology"] for row in groups}),
            median_selected_fingerprint_overlap=median(groups, "selected_fingerprint_overlap"),
            median_parent_feature_coverage=statistics.median([row[field] for row in groups for field in ["left_feature_coverage", "right_feature_coverage"]]),
            frame_fail_retained_count=len(frame_fail), frame_fail_but_joint_fit_below_0_5=sum(row["joint_fit_rmsd_A"] < .5 for row in frame_fail),
            frame_fail_but_both_sides_after_joint_fit_below_0_5=sum(row["joint_fit_max_side_rmsd_A"] < .5 for row in frame_fail),
            median_joint_fit_rmsd_A=median(retained, "joint_fit_rmsd_A"), median_parent_fit_rmsd_A=median(retained, "parent_fit_max_rmsd_A"),
            median_cross_distance_error_A=median(retained, "cross_distance_rms_error_A"),
            off_port_retained_candidates=sum(row["off_port_edges"] > 0 for row in retained),
            no_port_bridge_retained_candidates=sum(not row["port_bridge_exists"] for row in retained),
            connected_retained_without_port_bridge=sum(row["connected"] and not row["port_bridge_exists"] for row in retained),
            cross_parent_fixed_bond_candidates=sum(row["cross_parent_fixed_bonds"] > 0 for row in built),
            all_selected_feature_recovery_retained_count=sum(row["all_selected_features_recovered"] for row in retained),
            all_selected_atom_contact_recovery_retained_count=sum(row["all_selected_atom_contacts_recovered"] for row in retained),
            qualified_all_selected_features_count=sum(row["fusion_success"] and row["all_selected_features_recovered"] for row in built),
            qualified_with_selected_hbonds=sum(row["fusion_success"] and row["selected_hbond_features"] > 0 for row in built),
            qualified_all_selected_hbonds_recovered=sum(row["fusion_success"] and row["selected_hbond_features"] > 0 and row["all_selected_hbond_features_recovered"] for row in built),
            interaction_type_recovery=type_summary,
            omitted_shared_features=sorted({tuple(feature) for row in groups for feature in row["omitted_shared_features"]}),
            gly152_diagnostic={str(present):dict(count=sum(row["whole_GLY152_acceptor"] == present for row in built),
                mean_pic50=statistics.mean(row["pic50"] for row in built if row["whole_GLY152_acceptor"] == present) if any(row["whole_GLY152_acceptor"] == present for row in built) else None)
                for present in [True, False]},
            strata=strata, descriptive_spearman=correlations)
    assert len(group_rows) == 27 and len(terminal_rows) == 675
    table(output / "region_geometry.csv", group_rows)
    table(output / "terminal_geometry.csv", terminal_rows)
    save(output / "selection_details.json", details)
    packet = dict(schema="combmol-region-analysis-state-v1", parent_packet_id="evidence:" + evidence.manifest["archive_sha256"],
        provenance=dict(scientific_revision=evidence.manifest["scientific_revision"], coordinate_frame="unchanged archived protein frame, Angstrom",
                        representation="decoded final molecule and static source conditions", rdkit_version=rdBase.rdkitVersion,
                        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        correspondence="original source_atom to fixed slot; no terminal rematching"),
        observations=summary,
        coverage_limits=["No intermediate states are available in the compact evidence export.",
            "Exit directions follow removed source bonds and are descriptive, not a calibrated linker-feasibility test.",
            "No new affinity predictions, generation, relaxation or binding measurements were performed.",
            "Protein/ligand protonation was not independently validated; PLIP contacts are model-based evidence.",
            "Proper Kabsch fits here diagnose relative shape and do not replace the frozen protein-frame objective.",
            "Atom-exact contact matching may undercount chemically equivalent atom or contact-site alternatives.",
            "Off-port edges describe added chemistry outside source cut sites; the frozen protocol did not forbid them.",
            "The checkpoint's conditional-training tasks are not contained in the compact export and were not verified.",
            "GLY152 was inspected after observing the 6pyy selections; its score association is exploratory, not causal."],
        tables=["region_geometry.csv", "terminal_geometry.csv", "selection_details.json"])
    packet["table_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in packet["tables"]}
    packet["packet_id"] = "state:" + value_hash(packet)
    save(output / "StatePacket.json", packet)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
