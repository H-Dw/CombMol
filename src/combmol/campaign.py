"""Single-pass campaign with resumable evidence and write-ahead sampling slots."""
import concurrent.futures
import copy
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy
from rdkit import Chem
from .archive import index_target
from .evaluate import parent_profile_job, terminal_job, molecular_properties, summarize, table
from .geometry import clashes, protein_atoms, topology
from .io import digest, event, frozen, process_identity, read, value_hash, write, exclusive
from .ledger import AttemptLedger
from .native import NativeRuntime, verify_sources
from .provider import Provider, audit_screen, audit_fusions
from .regions import catalogue, chemical_graph, compile_condition, opportunities


def parallel_jobs(function, tasks, workers):
    if not tasks:
        return []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(function, tasks))


def prepare_parents(config, index, runtime, folder):
    folder = Path(folder)
    receipt = folder / "PARENTS.json"
    if receipt.exists():
        return read(receipt)["parents"]
    candidates = copy.deepcopy(index["candidates"])
    scores_path = folder / "SCORES.json"
    if scores_path.exists():
        scored = read(scores_path)
        if scored["sources"] != [candidate["evidence_id"] for candidate in candidates]:
            raise ValueError("Score cache source population differs")
        scores = scored["scores"]
    else:
        scores = []
        for start in range(0, len(candidates), 10):
            group = candidates[start:start + 10]
            molecules = [Chem.MolFromMolBlock(candidate["molblock"], removeHs=True) for candidate in group]
            scores.extend(runtime.score(molecules))
        frozen(scores_path, dict(sources=[candidate["evidence_id"] for candidate in candidates], scores=scores,
                                checkpoint_sha256=config["checkpoint_sha256"], protein_sha256=digest(index["protein"]),
                                policy="actual-discrete-endpoint-zero-noise-t0.9999-v1"))
    target_coordinates, target_elements = protein_atoms(index["protein"])
    tasks, basic_eligible = [], []
    rejected = []
    for candidate, score in zip(candidates, scores):
        molecule = Chem.MolFromMolBlock(candidate["molblock"], removeHs=True)
        candidate["pic50"] = score
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            rejected.append(dict(identifier=candidate["identifier"], reason="invalid_or_disconnected"))
            continue
        clash = clashes(molecule.GetConformer().GetPositions(), [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                         target_coordinates, target_elements, config["clash_ratio"])
        candidate["basic_quality"] = dict(protein_clash=clash, **molecular_properties(molecule))
        if clash["severe"]:
            rejected.append(dict(identifier=candidate["identifier"], reason="severe_protein_clash", measurement=clash))
            continue
        profile_folder = folder / "profiles" / candidate["record"]["node_id"]
        if (profile_folder / "PROFILE.json").exists() and (profile_folder / "QUALITY.json").exists():
            candidate["profile"] = read(profile_folder / "PROFILE.json")
            candidate["diagnostics"] = read(profile_folder / "QUALITY.json")
        else:
            tasks.append(dict(identifier=candidate["identifier"], source_evidence=candidate["evidence_id"],
                              molblock=candidate["molblock"], protein=index["protein"], folder=str(profile_folder)))
        basic_eligible.append(candidate)
    measured = {result["identifier"]: result for result in parallel_jobs(parent_profile_job, tasks, config["profile_workers"])}
    usable = []
    profile_errors = []
    for candidate in basic_eligible:
        result = measured.get(candidate["identifier"], candidate)
        if "profile_error" in result:
            profile_errors.append(result)
            continue
        candidate.update({name: result[name] for name in ("profile", "diagnostics")})
        if not candidate["profile"]["interactions"]:
            rejected.append(dict(identifier=candidate["identifier"], reason="no_real_PLIP_interactions"))
            continue
        candidate["graph"] = chemical_graph(Chem.MolFromMolBlock(candidate["molblock"], removeHs=True))
        usable.append(candidate)
    write(folder / "PREFILTER.json", dict(rejected=rejected, profile_errors=profile_errors,
                                          source_attempts=len(index["records"]), usable=len(usable)))
    if basic_eligible and not usable and profile_errors:
        raise RuntimeError("No usable interaction evidence; resolve saved PLIP errors before selection")
    provider = Provider(config)
    accepted = []
    for start in range(0, len(usable), config["chemical_review_batch"]):
        group = usable[start:start + config["chemical_review_batch"]]
        packet = dict(target=index["target"], candidates=[{name: candidate[name] for name in
            ("identifier", "evidence_id", "pic50", "graph", "profile", "basic_quality")} for candidate in group])
        result = provider.ask("screen", packet, folder / "llm_screen" / f"batch_{start:03d}",
                              lambda result: audit_screen(result, group))
        kept = {decision["candidate_identifier"] for decision in result["decisions"] if decision["keep"]}
        accepted.extend(candidate for candidate in group if candidate["identifier"] in kept)
    accepted.sort(key=lambda candidate: (-candidate["pic50"], candidate["identifier"]))
    parents, topologies = [], set()
    for candidate in accepted:
        graph_key = topology(Chem.MolFromMolBlock(candidate["molblock"], removeHs=True))
        if graph_key not in topologies:
            topologies.add(graph_key)
            parents.append(candidate)
        if len(parents) == config["max_parents"]:
            break
    frozen(receipt, dict(parents=parents, source_attempts=len(index["records"]),
                         accepted_poses=len(accepted), eligible_poses=len(usable)))
    table(folder / "parent_scores.csv", [dict(identifier=parent["identifier"], pic50=parent["pic50"],
          historical_pic50=parent["record"]["pic50_on_rescore"], **parent["basic_quality"]) for parent in parents])
    return parents


def choose_conditions(config, target, parents, folder):
    folder = Path(folder)
    path = folder / "CONDITIONS.json"
    if path.exists():
        return read(path)
    regions = [region for parent in parents for region in catalogue(parent, config["repository"])]
    frozen(folder / "REGIONS.json", regions)
    if len(parents) < 2 or not regions:
        return frozen(path, [])
    examples = opportunities(parents, regions, config)
    frozen(folder / "OPPORTUNITIES.json", examples)
    packet = dict(target=target, parents=[{name: parent[name] for name in
        ("identifier", "evidence_id", "pic50", "graph", "profile", "basic_quality")} for parent in parents],
        regions=regions, pair_opportunities=examples, max_proposals=16,
        coordinate_frame="shared unchanged protein frame; no ligand alignment",
        clash_ratio=config["clash_ratio"], final_inpaint=False)
    result = Provider(config).ask("fusion", packet, folder / "llm_fusion",
                                  lambda result: audit_fusions(result, parents, regions, config))
    conditions = [compile_condition(proposal, parents, regions, config) for proposal in result["proposals"]]
    if len(conditions) > 16:
        raise ValueError("Too many conditions")
    return frozen(path, conditions)


def run_condition(config, condition, parents, index, runtime, folder, revision):
    folder = Path(folder)
    frozen(folder / "CONDITION.json", condition)
    ledger = AttemptLedger(folder / "ATTEMPTS.json", condition["condition_hash"], config["seeds"], config["replicates"])
    parent_scores = [parent["pic50"] for parent in parents if parent["identifier"] in
                     {contribution["parent_identifier"] for contribution in condition["proposal"]["parent_regions"]}]
    for seed in config["seeds"]:
        batch_folder = folder / f"seed_{seed}"
        terminal_path = batch_folder / "TERMINALS.pt"
        records_path = batch_folder / "TERMINAL_RECORDS.json"
        pending = ledger.pending(seed)
        if pending:
            if len(pending) != config["replicates"]:
                raise ValueError("Partial seed reservation requires explicit diagnosis")
            batch = runtime.prepare_generation(condition, seed)
            batch_folder.mkdir(parents=True, exist_ok=True)
            torch = runtime.torch
            torch.save({"prior": {name: tensor.detach().cpu() for name, tensor in batch["prior"].items() if torch.is_tensor(tensor)},
                        "torch_rng": torch.get_rng_state(), "accelerator_rng": torch.cuda.get_rng_state_all()}, batch_folder / "INITIALIZATION.pt")
            frozen(batch_folder / "INITIALIZATION.json", dict(hash=batch["initialization_hash"], atom_count=batch["atom_count"], seed=seed))
            ledger.start(pending, batch["initialization_hash"], revision)
            try:
                tensors, molecules = runtime.generate(batch)
                temporary = terminal_path.with_suffix(".tmp")
                torch.save(tensors, temporary)
                os.replace(temporary, terminal_path)
                rows = []
                for replicate, molecule in enumerate(molecules):
                    node = f"{condition['proposal']['identifier']}_seed{seed}_rep{replicate}"
                    molblock = Chem.MolToMolBlock(molecule) if molecule is not None else None
                    if molecule is not None:
                        molecule.SetProp("node_id", node)
                        with Chem.SDWriter(str(batch_folder / f"replicate_{replicate}.sdf")) as writer:
                            writer.write(molecule)
                    rows.append(dict(node_id=node, seed=seed, replicate=replicate, molblock=molblock))
                frozen(records_path, rows)
                for key in pending:
                    ledger.finish(key, terminal_file=str(terminal_path), terminal_sha256=digest(terminal_path))
            except Exception as error:
                ledger.fail_started(type(error).__name__ + ": " + str(error)[:1800])
                raise
        if not records_path.exists():
            continue
        rows = read(records_path)
        scores_path = batch_folder / "ENDPOINT_SCORES.json"
        if scores_path.exists():
            scores = read(scores_path)
        else:
            usable = [row for row in rows if row["molblock"]]
            measured = runtime.score([Chem.MolFromMolBlock(row["molblock"], removeHs=True) for row in usable], seed)
            scores = {row["node_id"]: score for row, score in zip(usable, measured)}
            frozen(scores_path, scores)
        tasks = []
        for row in rows:
            tasks.append(dict(**row, folder=str(batch_folder / row["node_id"]), condition=condition,
                protein=index["protein"], pic50=scores.get(row["node_id"]), clash_ratio=config["clash_ratio"],
                rmsd_limit_A=config["condition_rmsd_limit_A"], binding=value_hash(dict(condition=condition["condition_hash"],
                    molblock=row["molblock"], score=scores.get(row["node_id"]), protein=digest(index["protein"]), protocol=config["protocol"]))))
        parallel_jobs(terminal_job, tasks, config["profile_workers"])
    rows = [read(path) for path in sorted(folder.glob("seed_*/*/EVALUATION.json"))]
    # Started slots without a terminal still occupy their original scientific quota.
    completed_keys = {(row["seed"], row["replicate"]) for row in rows}
    for slot in ledger.data["slots"].values():
        if slot["state"] != "pending" and (slot["seed"], slot["replicate"]) not in completed_keys:
            rows.append(dict(seed=slot["seed"], replicate=slot["replicate"], build_success=False,
                             fusion_success=False, pic50=None, failure_reason=slot.get("reason", "started_slot_without_terminal")))
    if len(rows) > 25:
        raise ValueError("Condition exceeded attempt cap")
    summary = summarize(rows, parent_scores)
    summary.update(condition_hash=condition["condition_hash"], pending=25-ledger.spent(),
                   failed_slots=sum(slot["state"] in ("failed", "started") for slot in ledger.data["slots"].values()))
    table(folder / "candidates.csv", rows)
    write(folder / "SUMMARY.json", summary)
    return summary


def run(config, attempt="attempt_001"):
    root = Path(config["result_root"])
    root.mkdir(parents=True, exist_ok=True)
    if config.get("plip_dependencies") not in sys.path:
        sys.path.append(config["plip_dependencies"])
    with exclusive(root / "CAMPAIGN.lock"):
        revision = subprocess.check_output(["git", "-C", config["repository"], "rev-parse", "HEAD"], text=True).strip()
        frozen(root / "CONFIG.json", {name: value for name, value in config.items() if name != "repository"})
        run_folder = root / "runs" / attempt
        frozen(run_folder / "LAUNCH.json", dict(**process_identity(), revision=revision, repository=config["repository"]))
        write(root / "LAUNCH.json", read(run_folder / "LAUNCH.json"))
        event(root, "runtime_preflight", revision=revision, attempt=attempt)
        verify_sources(config)
        frozen(run_folder / "RUNTIME.json", dict(revision=revision, protocol=config["protocol"],
            checkpoint_sha256=config["checkpoint_sha256"], upstream_commit=config["upstream_commit"],
            apply_guidance=False, final_inpaint=False))
        results = []
        for target in config["targets"]:
            destination = root / "targets" / target
            try:
                if (destination / "FINISHED.json").exists():
                    previous = read(destination / "FINISHED.json")
                    if previous["complete"]:
                        results.append(previous)
                        continue
                event(root, "source_index", target=target, attempt=attempt)
                index = index_target(config, target, destination / "sources")
                event(root, "parent_scoring_and_profiles", target=target, attempt=attempt)
                runtime = NativeRuntime(config, index["protein"], index["reference_ligand"], destination / "native")
                frozen(destination / "native/NATIVE_SETTINGS.json", dict(arguments=vars(runtime.args),
                    model_inpainting_mode=runtime.model.inpainting_mode,
                    model_inpainting_mode_inf=runtime.model.inpainting_mode_inf,
                    coord_scale=runtime.model.coord_scale,
                    use_inpaint_self_condition=runtime.model._inpaint_self_condition))
                parents = prepare_parents(config, index, runtime, destination / "selection")
                event(root, "fragment_combination", target=target, attempt=attempt)
                conditions = choose_conditions(config, target, parents, destination / "selection")
                summaries = []
                for condition in conditions:
                    event(root, "native_inpainting_and_evaluation", target=target,
                          condition=condition["proposal"]["identifier"], attempt=attempt)
                    summaries.append(run_condition(config, condition, parents, index, runtime,
                        destination / "fusions" / condition["proposal"]["identifier"], revision))
                target_result = dict(target=target, status="complete" if conditions else "no_feasible_combination",
                    parents=len(parents), conditions=len(conditions), attempts=sum(row["attempts"] for row in summaries),
                    successful_fusions=sum(row["successful_fusions"] for row in summaries), summaries=summaries,
                    complete=all(row["attempts"] == 25 and row["pending"] == 0 and row["failed_slots"] == 0
                                 and row["evaluation_errors"] == 0 for row in summaries))
                if target_result["attempts"] > 400:
                    raise ValueError("Target budget exceeded")
                write(destination / "FINISHED.json", target_result)
                table(destination / "fusion_summary.csv", summaries)
                results.append(target_result)
                del runtime
                import torch
                torch.cuda.empty_cache()
            except Exception as error:
                failure = dict(target=target, status="failed", complete=False, error_type=type(error).__name__, message=str(error)[:4000])
                write(run_folder / f"{target}_FAILED.json", dict(**failure, traceback=traceback.format_exc()))
                write(destination / "FAILED.json", failure)
                results.append(failure)
        spent = sum(sum(slot["state"] != "pending" for slot in read(path)["slots"].values())
                    for path in root.glob("targets/*/fusions/*/ATTEMPTS.json"))
        if spent > 800:
            raise ValueError("Campaign scientific budget exceeded")
        report = dict(protocol=config["protocol"], revision=revision, targets=results, spent_attempts=spent,
                      upper_bound=800, all_success=all(row["complete"] for row in results),
                      no_gradient_guidance=True, final_inpaint=False)
        write(run_folder / "FINISHED.json", report)
        write(root / "FINISHED.json", report)
        write(root / "ONLINE_PROTOCOL_AUDIT.json", dict(spent_attempts=spent, upper_bound=800,
            unique_slot_accounting=True, apply_guidance=False, final_inpaint=False,
            seeds=config["seeds"], replicates=5, max_fusions_per_target=16, max_parents_per_target=8,
            checkpoint_sha256=config["checkpoint_sha256"], complete=report["all_success"]))
        from .report import campaign_report
        campaign_report(root, report)
        event(root, "finished", all_success=report["all_success"], spent_attempts=spent)
        return report
