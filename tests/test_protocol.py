import copy
import io
import json
import math
import tarfile
from pathlib import Path
import numpy
import pytest
from rdkit import Chem
from combmol.archive import index_target
from combmol.config import load, SEEDS
from combmol.geometry import clashes, frame_rmsd, proper_alignment, size_range, topology
from combmol.io import read, value_hash, write, frozen
from combmol.ledger import AttemptLedger
from combmol.provider import Provider, audit_screen, audit_fusions
from combmol.regions import catalogue, chemical_graph, compile_condition, protected_units, region_valid
from combmol.evaluate import summarize

REPOSITORY = Path(__file__).resolve().parents[1]


def molecule(smiles, offset=0):
    result = Chem.MolFromSmiles(smiles)
    conformer = Chem.Conformer(result.GetNumAtoms())
    conformer.Set3D(True)
    for atom_index in range(result.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, (float(atom_index * 1.5 + offset), float(atom_index % 2), float(atom_index % 3)))
    result.AddConformer(conformer)
    return result


def parent(smiles, identifier, offset=0):
    source = molecule(smiles, offset)
    contact_atom = source.GetNumAtoms() - 1
    contact = dict(ligand_atoms=[contact_atom], evidence_id="contact:" + identifier,
                   fingerprint=["hbond", "A", "1", "ASP", "ligand_acceptor"])
    return dict(identifier=identifier, evidence_id="source:" + identifier,
                molblock=Chem.MolToMolBlock(source), source_member_sha256="immutable-source",
                profile=dict(interactions=[contact]), graph=chemical_graph(source))


def test_frozen_config_and_source_snapshot():
    config = load(REPOSITORY / "configs/campaign.json")
    assert len(config["seeds"]) * config["replicates"] * config["max_fusions"] * len(config["targets"]) == 800
    from combmol.io import digest
    upstream = read(REPOSITORY / "third_party/UPSTREAM.json")
    assert all(digest(REPOSITORY / "third_party/flowr_root" / name) == checksum for name, checksum in upstream["files"].items())
    assert not list((REPOSITORY / "third_party").rglob("*.ckpt"))


def test_pseudocode_precedes_implementation():
    receipt = read(REPOSITORY / "docs/pseudocode/PRECODE_CHECK.json")
    assert receipt["created_before_implementation"] and receipt["implementation_files_present"] == 0
    assert len(receipt["modules"]) == 13


def test_topology_ignores_conformation_order_and_stereo():
    source = molecule("C[C@H](O)CC")
    renumbered = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    assert topology(source) == topology(renumbered) == topology(molecule("C[C@@H](O)CC"))
    assert topology(source) != topology(molecule("CCC(=O)C"))


def test_geometry_shared_frame_and_reflection():
    points = numpy.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    rotation = numpy.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = points @ rotation + [5, 3, -2]
    fitted_rotation, translation = proper_alignment(moved, points)
    assert frame_rmsd(moved @ fitted_rotation + translation, points) < 1e-12
    assert frame_rmsd(moved, points) > 1
    reflected = points * [-1, 1, 1]
    fitted_rotation, translation = proper_alignment(reflected, points)
    assert numpy.linalg.det(fitted_rotation) > 0
    assert frame_rmsd(reflected @ fitted_rotation + translation, points) > 0.1
    with pytest.raises(ValueError):
        proper_alignment(points * numpy.nan, points)


def test_clash_is_element_aware_and_rigid_invariant():
    assert clashes([[0, 0, 0]], [6], [[1, 0, 0]], [6])["severe"]
    assert not clashes([[0, 0, 0]], [6], [[4, 0, 0]], [6])["severe"]
    assert clashes([[5, 2, 1]], [6], [[6, 2, 1]], [6])["min_ratio"] == clashes([[0, 0, 0]], [6], [[1, 0, 0]], [6])["min_ratio"]


@pytest.mark.parametrize("fixed_count,boundary_distance", [(4, 3), (18, 8), (29, 12)])
def test_size_range_includes_free_atoms(fixed_count, boundary_distance):
    limits = size_range([17, 22], fixed_count, boundary_distance)
    assert limits["lower"] >= fixed_count + limits["minimum_free"]
    assert limits["upper"] >= limits["lower"]
    with pytest.raises(ValueError):
        size_range([17, 22], 250, 8)


def test_ledger_never_reuses_started_slots(tmp_path):
    ledger = AttemptLedger(tmp_path / "ledger.json", "condition", SEEDS)
    slots = ledger.pending(SEEDS[0])
    ledger.start(slots, "initialization", "revision")
    assert ledger.spent() == 5
    with pytest.raises(ValueError):
        ledger.start(slots, "new", "new")
    ledger.finish(slots[0], terminal="saved")
    ledger.fail_started("failed sampler")
    restored = AttemptLedger(tmp_path / "ledger.json", "condition", SEEDS)
    assert restored.spent() == 5 and not restored.pending(SEEDS[0])
    assert len(restored.pending(SEEDS[1])) == 5
    with pytest.raises(ValueError):
        AttemptLedger(tmp_path / "ledger.json", "other-condition", SEEDS)


def test_ledger_reservation_is_all_or_none(tmp_path):
    ledger = AttemptLedger(tmp_path / "ledger.json", "condition", SEEDS)
    with pytest.raises(KeyError):
        ledger.start(["missing"], "hash", "revision")
    assert ledger.spent() == 0


def test_ring_and_functional_group_integrity():
    source = molecule("CCC(=O)NCCc1ccccc1")
    units = protected_units(source, REPOSITORY)
    assert not region_valid(source, [2, 3], units)
    assert not region_valid(source, [7, 8, 9], units)
    assert region_valid(source, [0], units)
    assert not region_valid(source, range(source.GetNumAtoms()), units)


def test_catalogue_uses_real_contacts_and_proper_subsets():
    packet = parent("CCCO", "parent-left")
    regions = catalogue(packet, REPOSITORY)
    assert regions
    assert all(3 in region["atom_indices"] and len(region["atom_indices"]) >= 2 for region in regions)
    assert all(len(region["atom_indices"]) < 4 for region in regions)
    assert all(region["interaction_evidence"] == ["contact:parent-left"] for region in regions)


def test_condition_contains_no_invented_parent_connection():
    config = load(REPOSITORY / "configs/campaign.json")
    parents = [parent("CCCO", "parent-left"), parent("CCCCN", "parent-right", 10)]
    regions = [region for packet in parents for region in catalogue(packet, REPOSITORY)]
    chosen = [next(region for region in regions if region["parent"] == packet["identifier"]) for packet in parents]
    proposal = dict(identifier="fusion-01", parent_regions=[dict(parent_identifier=region["parent"], region_identifiers=[region["identifier"]]) for region in chosen])
    condition = compile_condition(proposal, parents, regions, config)
    fixed = Chem.MolFromMolBlock(condition["fixed_molblock"], removeHs=True)
    assert len(Chem.GetMolFrags(fixed)) == 2
    assert set(source["parent_identifier"] for source in condition["sources"]) == {"parent-left", "parent-right"}
    assert not condition["final_inpaint"] and not condition["apply_guidance"]


def test_screen_requires_complete_identity_and_bound_evidence():
    candidates = [dict(identifier="one", evidence_id="source:one")]
    audit_screen(dict(decisions=[dict(candidate_identifier="one", evidence_ids=["source:one"])]), candidates)
    with pytest.raises(ValueError):
        audit_screen(dict(decisions=[]), candidates)
    with pytest.raises(ValueError):
        audit_screen(dict(decisions=[dict(candidate_identifier="one", evidence_ids=["invented"])]), candidates)


def test_provider_bounds_schema_repair_and_reuses_receipts(tmp_path, monkeypatch):
    config = load(REPOSITORY / "configs/campaign.json")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-only")
    replies = [dict(decisions=[]), dict(decisions=[dict(candidate_identifier="one", keep=True, evidence_ids=["source:one"], rationale="Intact functional group")])]
    calls = []
    def respond(*args, **kwargs):
        calls.append(kwargs["json"])
        content = replies[len(calls) - 1]
        return type("Reply", (), dict(status_code=200, json=lambda self: dict(model="deepseek-flash", choices=[dict(finish_reason="stop", message=dict(content=json.dumps(content)))])))()
    monkeypatch.setattr("combmol.provider.requests.post", respond)
    candidates = [dict(identifier="one", evidence_id="source:one")]
    provider = Provider(config)
    result = provider.ask("screen", dict(candidates=candidates), tmp_path / "request", lambda result: audit_screen(result, candidates))
    assert len(calls) == 2 and result["decisions"][0]["keep"]
    provider.ask("screen", dict(candidates=candidates), tmp_path / "request", lambda result: audit_screen(result, candidates))
    assert len(calls) == 2
    assert "unit-test-only" not in "".join(path.read_text() for path in tmp_path.rglob("*.json"))


def test_failed_attempts_are_not_high_score_successes():
    rows = [dict(build_success=True, fusion_success=True, pic50=7.),
            dict(build_success=False, fusion_success=False, pic50=99.),
            dict(build_success=True, fusion_success=False, pic50=20.)]
    summary = summarize(rows, [6., 6.5])
    assert summary["attempts"] == 3 and summary["successful_fusions"] == 1
    assert summary["valid_fusion_mean_pic50"] == 7.


def test_archive_indexes_node_property_instead_of_record_order(tmp_path):
    target = "test_target"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    records, blocks = [], []
    for position in range(100):
        source = molecule("CCO")
        identifier = f"source_{position:03d}"
        source.SetProp("node_id", identifier)
        source.SetProp("pic50_on_rescore", str(float(position)))
        records.append(dict(node_id=identifier, build_success=True, pic50_on_rescore=float(position)))
        output = io.StringIO()
        writer = Chem.SDWriter(output)
        writer.write(source)
        writer.close()
        blocks.append(output.getvalue())
    entries = {f"targets/{target}/results/unguided_w050/final_records.json": json.dumps(list(reversed(records))).encode(),
               f"targets/{target}/results/unguided_w050/unguided/batch_000/molecules_all_built.sdf": "".join(blocks).encode(),
               f"targets/{target}/inputs/target_protein.pdb": b"ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
               f"targets/{target}/inputs/target_ligand.sdf": blocks[0].encode()}
    archive = dataset / "targets.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, payload in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    write(dataset / "results_summary.json", dict(targets={target: dict(archive="targets.tar.gz")}))
    result = index_target(dict(dataset_root=str(dataset)), target, tmp_path / "source")
    assert result["candidates"][0]["identifier"] == "test_target/source_099"
    assert result["candidates"][0]["record"]["pic50_on_rescore"] == 99.
    assert len(result["candidates"]) == 100


def test_native_feature_contract_with_real_torch_when_available():
    torch = pytest.importorskip("torch")
    from combmol.native import validate_features
    values = dict(coords=torch.zeros(2, 4, 3), bonds=torch.zeros(2, 4, 4, 5), mask=torch.ones(2, 4, dtype=torch.long))
    validate_features(values, values, torch)
    with pytest.raises(ValueError):
        validate_features(dict(values, mask=torch.ones(2, 4)), values, torch)
    with pytest.raises(ValueError):
        validate_features(dict(values, bonds=torch.zeros(2, 4, 3)), values, torch)


def test_native_runner_only_calls_unguided_inpainting():
    torch = pytest.importorskip("torch")
    from combmol.native import NativeRuntime
    calls = []
    runtime = NativeRuntime.__new__(NativeRuntime)
    runtime.torch = torch
    runtime.device = torch.device("cpu")
    runtime.config = dict(batch_size=50)
    class Model:
        def _generate(self, *args, **kwargs):
            calls.append(kwargs)
            return dict(coords=torch.zeros(5, 4, 3), mask=torch.ones(5, 4, dtype=torch.long))
        def _generate_mols(self, output, **kwargs):
            assert kwargs["valence_repair"] is False
            return [None] * 5
    runtime.model = Model()
    tensors, molecules = runtime.generate(dict(prior=dict(coords=torch.zeros(5, 4, 3)), pocket={}))
    assert len(molecules) == 5
    assert calls[0]["apply_guidance"] is False and calls[0]["final_inpaint"] is False
    assert calls[0]["steps"] == 100 and calls[0]["solver"] == "euler"
