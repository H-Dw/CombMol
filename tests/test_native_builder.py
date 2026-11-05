"""CPU checks using the actual pinned native builder, without generating candidates."""
import sys
from pathlib import Path
import pytest


def test_runtime_supplies_the_real_native_parser_contract(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "third_party/flowr_root"))
    from flowr.gen.generate_from_pdb import get_args
    from combmol.native import NativeRuntime
    from combmol.config import load
    parsed = []
    class ParserReached(Exception):
        pass
    def inspect_then_stop():
        parsed.append(get_args())
        raise ParserReached
    monkeypatch.setattr("flowr.gen.generate_from_pdb.get_args", inspect_then_stop)
    with pytest.raises(ParserReached):
        NativeRuntime(load(root / "configs/campaign.json"), "protein.pdb", "ligand.sdf", tmp_path)
    assert parsed[0].arch == "pocket" and parsed[0].pocket_type == "holo"
    assert parsed[0].fragment_growing and not parsed[0].ligand_valence_repair


def test_real_native_builder_keeps_fixed_coordinates_and_nonbonds():
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "third_party/flowr_root"))
    from flowr.models.mol_builder import MolBuilder
    builder = MolBuilder.__new__(MolBuilder)
    prior = dict(coords=torch.arange(12).reshape(1, 4, 3).float(),
                 atomics=torch.zeros(1, 4, 5), bonds=torch.zeros(1, 4, 4, 5),
                 charges=torch.zeros(1, 4, 3), mask=torch.ones(1, 4, dtype=torch.long),
                 fragment_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.bool), fragment_mode=["fragment_growing"])
    prior["bonds"][..., 0] = 1
    prediction = dict(coords=torch.full((1, 4, 3), 100.), atomics=torch.ones(1, 4, 5),
                      bonds=torch.zeros(1, 4, 4, 5), charges=torch.ones(1, 4, 3))
    prediction["bonds"][..., 1] = 1
    result = builder.inpaint_molecule(prior, prediction)
    assert torch.equal(result["coords"][0, :2], prior["coords"][0, :2])
    assert torch.equal(result["coords"][0, 2:], torch.full((2, 3), 100.))
    assert result["bonds"][0, 0, 1, 0] == 1
    assert result["bonds"][0, 0, 2, 1] == 1
