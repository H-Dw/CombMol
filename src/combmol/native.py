"""Thin adapter around the pinned native features, interpolant and sampler."""
import copy
import hashlib
import os
import random
import sys
from pathlib import Path
import numpy
from rdkit import Chem
from .io import digest, read, value_hash


def verify_sources(config):
    repository = Path(config["repository"])
    upstream = read(repository / "third_party/UPSTREAM.json")
    if upstream["commit"] != config["upstream_commit"]:
        raise ValueError("Unexpected FLOWR release")
    for relative, checksum in upstream["files"].items():
        if digest(repository / "third_party/flowr_root" / relative) != checksum:
            raise ValueError("Vendored source changed: " + relative)
    if digest(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint identity differs")
    return upstream


def tensor_hash(tensors):
    checksum = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        if hasattr(tensor, "detach"):
            array = tensor.detach().cpu().contiguous().numpy()
            checksum.update(name.encode())
            checksum.update(str((array.shape, array.dtype)).encode())
            checksum.update(array.tobytes())
    return checksum.hexdigest()


def validate_features(ligand, pocket, torch):
    for values in (ligand, pocket):
        if values["coords"].ndim != 3 or values["bonds"].ndim != 4:
            raise ValueError("Native coordinates or dense bond features have incorrect rank")
        batch_count, atom_count = values["coords"].shape[:2]
        if values["bonds"].shape[:3] != (batch_count, atom_count, atom_count):
            raise ValueError("Dense adjacency dimensions differ")
        for name in ("mask", "atom_names", "res_names"):
            if name in values and values[name].dtype not in (torch.bool, torch.int32, torch.int64):
                raise ValueError("Index or mask tensor must be integer/bool: " + name)
        for name, tensor in values.items():
            if torch.is_tensor(tensor):
                if tensor.device != values["coords"].device:
                    raise ValueError("Mixed devices in native features: " + name)
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError("Nonfinite native features: " + name)


class NativeRuntime:
    def __init__(self, config, protein, reference_ligand, output):
        self.config = config
        upstream = Path(config["repository"]) / "third_party/flowr_root"
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        # Only the hash-verified trusted research checkpoint is unpickled.
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        import torch
        from flowr.gen.generate_from_pdb import get_args
        from flowr.gen import utils
        from flowr.scriptutil import load_model
        self.torch, self.utils = torch, utils
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("high")
        previous = sys.argv
        try:
            sys.argv = ["CombMol", "--pdb_file", str(protein), "--ligand_file", str(reference_ligand),
                        "--ckpt_path", config["checkpoint"], "--save_dir", str(output),
                        "--batch_cost", str(config["batch_size"]), "--num_workers", "0",
                        "--cut_pocket", "--pocket_cutoff", str(config["pocket_cutoff_A"]),
                        "--fragment_growing", "--grow_size", "4", "--no_ligand_valence_repair"]
            self.args = get_args()
        finally:
            sys.argv = previous
        self.args.coord_noise_scale = 0.0
        self.args.rotation_alignment = False
        self.args.permutation_alignment = False
        self.args.final_inpaint = False
        self.args.sample_mol_sizes = False
        self.args.use_sde_simulation = False
        self.args.corrector_iters = 0
        (self.model, self.hparams, self.vocab, self.vocab_charges, self.vocab_hybridization,
         self.vocab_aromatic, _, _) = load_model(self.args)
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("No accelerator available for the campaign")
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.source_system = utils.load_data_from_pdb(
            self.args, remove_hs=self.hparams["remove_hs"], remove_aromaticity=self.hparams["remove_aromaticity"],
            canonicalize_conformer=False)
        if self.source_system.com is not None:
            raise ValueError("Expected an uncentered native source system")

    def batch(self, molecules, seed, free_atoms=None):
        from flowr.util.molrepr import GeometricMol
        from flowr.data.datamodules import GeometricInterpolantDM
        torch = self.torch
        random.seed(seed)
        numpy.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        args = copy.copy(self.args)
        args.fragment_growing = free_atoms is not None
        args.grow_size = free_atoms
        transform, interpolant = self.utils.load_util(args, self.hparams, self.vocab,
            self.vocab_charges, self.vocab_hybridization, self.vocab_aromatic)
        systems = []
        for molecule in molecules:
            geometric = GeometricMol.from_rdkit(Chem.Mol(molecule), kekulize=self.hparams["remove_aromaticity"])
            system = self.source_system._copy_with(ligand=geometric)
            transformed = transform(system)
            recovered = transformed.ligand.coords.cpu().numpy() + transformed.com.cpu().numpy()
            if not numpy.allclose(recovered, molecule.GetConformer().GetPositions(), atol=1e-4, rtol=0):
                raise ValueError("Native featurization moved or reordered source heavy atoms")
            systems.append(transformed)
        module = GeometricInterpolantDM(None, None, None, 50, 50, test_interpolant=interpolant,
                                       num_workers=0, pad_to_bucket=False)
        prior, posterior, _, _ = module._collate(systems, "test")
        ligand_prior = self.model.builder.extract_ligand_from_complex(prior)
        ligand_data = self.model.builder.extract_ligand_from_complex(posterior)
        pocket = self.model.builder.extract_pocket_from_complex(posterior)
        for values, original in ((ligand_prior, prior), (ligand_data, posterior)):
            values["interactions"] = original["interactions"]
            values["fragment_mask"] = original["fragment_mask"]
            values["fragment_mode"] = original["fragment_mode"]
        pocket["interactions"] = posterior["interactions"]
        pocket["complex"] = posterior["complex"]
        for values in (ligand_prior, ligand_data, pocket):
            for name, tensor in values.items():
                if torch.is_tensor(tensor):
                    values[name] = tensor.to(self.device)
            values["mask"] = values["mask"].long()
        validate_features(ligand_prior, pocket, torch)
        validate_features(ligand_data, pocket, torch)
        if free_atoms is not None:
            for row in range(len(molecules)):
                fixed_count = int(ligand_prior["fragment_mask"][row].sum())
                if fixed_count != molecules[row].GetNumAtoms() or not 0 < fixed_count < int(ligand_prior["mask"][row].sum()):
                    raise ValueError("Empty condition or missing free region; de novo fallback forbidden")
        return dict(prior=ligand_prior, posterior=ligand_data, pocket=pocket)

    def score(self, molecules, seed=42):
        if not molecules:
            return []
        torch = self.torch
        batch = self.batch(molecules, seed)
        count = len(molecules)
        times = [torch.full((count,), 0.9999, device=self.device),
                 torch.full((count,), 0.9999, device=self.device), torch.ones(count, device=self.device)]
        with torch.no_grad():
            result = self.model._predict_affinity(batch["prior"], batch["posterior"], batch["pocket"], times)
        scores = result["affinity"]["pic50"].detach().cpu().reshape(-1).tolist()
        if len(scores) != count or not numpy.isfinite(scores).all():
            raise ValueError("Incomplete or nonfinite endpoint scores")
        return [float(score) for score in scores]

    def prepare_generation(self, condition, seed):
        molecule = Chem.MolFromMolBlock(condition["fixed_molblock"], removeHs=True)
        sizes = condition["size_range"]
        atom_count = int(numpy.random.default_rng(seed).integers(sizes["lower"], sizes["upper"] + 1))
        batch = self.batch([molecule] * self.config["replicates"], seed, free_atoms=atom_count - molecule.GetNumAtoms())
        # An input-only check uses the exact fixed slots, not an alignment that could hide pose errors.
        fixed_count = molecule.GetNumAtoms()
        for row in range(self.config["replicates"]):
            system = batch["pocket"]["complex"][row]
            restored = batch["prior"]["coords"][row, :fixed_count].detach().cpu().numpy() * self.model.coord_scale + system.com.cpu().numpy()
            if not numpy.allclose(restored, molecule.GetConformer().GetPositions(), atol=1e-4, rtol=0):
                raise ValueError("Native initial condition does not preserve source coordinates")
        batch["initialization_hash"] = tensor_hash(batch["prior"])
        batch["atom_count"] = atom_count
        return batch

    def generate(self, batch):
        torch = self.torch
        count = batch["prior"]["coords"].shape[0]
        if not 0 < count <= self.config["batch_size"]:
            raise ValueError("Batch exceeds the declared sample cap")
        times = [torch.zeros(count, device=self.device) for _ in range(3)]
        with torch.no_grad():
            output = self.model._generate(batch["prior"], batch["pocket"], steps=100, times=times,
                strategy="linear", solver="euler", corr_iters=0, save_traj=False,
                apply_guidance=False, final_inpaint=False, coord_noise_level=0.0)
        tensors = {name: tensor.detach().cpu() for name, tensor in output.items() if torch.is_tensor(tensor)}
        molecules = self.model._generate_mols(output, valence_repair=False)
        if len(molecules) != count:
            raise ValueError("Native decode lost slot identities")
        return tensors, molecules
