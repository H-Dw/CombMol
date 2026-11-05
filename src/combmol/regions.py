"""Source-only regions, protected chemistry, and deterministic condition assembly."""
import importlib.util
import itertools
from pathlib import Path
import numpy
from rdkit import Chem
from .geometry import clashes, size_range, topology
from .io import value_hash


def chemical_graph(molecule):
    positions = molecule.GetConformer().GetPositions()
    return dict(smiles=Chem.MolToSmiles(molecule, isomericSmiles=True),
                atoms=[dict(atom_index=atom.GetIdx(), element=atom.GetSymbol(), atomic_number=atom.GetAtomicNum(),
                            formal_charge=atom.GetFormalCharge(), hybridization=str(atom.GetHybridization()),
                            aromatic=atom.GetIsAromatic(), coordinates_A=positions[atom.GetIdx()].tolist())
                       for atom in molecule.GetAtoms()],
                bonds=[dict(begin=bond.GetBeginAtomIdx(), end=bond.GetEndAtomIdx(),
                            order=bond.GetBondTypeAsDouble(), aromatic=bond.GetIsAromatic(),
                            conjugated=bond.GetIsConjugated(), ring=bond.IsInRing()) for bond in molecule.GetBonds()])


def protected_units(molecule, repository):
    source = Path(repository) / "third_party/flowr_root/flowr/util/ifg.py"
    spec = importlib.util.spec_from_file_location("combmol_upstream_functional_groups", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    units = [set(group.atomIds) for group in module.identify_functional_groups(molecule)]
    units.extend(set(ring) for ring in molecule.GetRingInfo().AtomRings())
    # Keep conjugated amide/urea/sulfonamide units intact even when a pattern spans a cut bond.
    for pattern in ["[N]-[C](=[O,S])", "[N]-[S](=[O])(=[O])", "[O]-[P](=[O])", "[N]-[C](=[N])-[N]"]:
        units.extend(set(match) for match in molecule.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
    return units


def region_valid(molecule, atom_set, units):
    atom_set = set(atom_set)
    if not atom_set or len(atom_set) >= molecule.GetNumAtoms():
        return False
    if any(atom_set & unit and not unit <= atom_set for unit in units):
        return False
    boundaries = [bond for bond in molecule.GetBonds()
                  if (bond.GetBeginAtomIdx() in atom_set) != (bond.GetEndAtomIdx() in atom_set)]
    return bool(boundaries) and all(not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE for bond in boundaries)


def catalogue(packet, repository):
    molecule = Chem.MolFromMolBlock(packet["molblock"], removeHs=True)
    units = protected_units(molecule, repository)
    protected_bonds = {bond.GetIdx() for bond in molecule.GetBonds() if any(
        {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= unit for unit in units)}
    cuttable = [bond.GetIdx() for bond in molecule.GetBonds()
                if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE
                and bond.GetIdx() not in protected_bonds]
    atom_sets = set()
    for cut_count in (1, 2):
        for cuts in itertools.combinations(cuttable, cut_count):
            fragmented = Chem.FragmentOnBonds(molecule, list(cuts), addDummies=False)
            atom_sets.update(tuple(sorted(component)) for component in Chem.GetMolFrags(fragmented))
    interactions = packet["profile"]["interactions"]
    result = []
    for atoms in sorted(atom_sets):
        selected = set(atoms)
        contacts = [contact for contact in interactions if set(contact["ligand_atoms"]) <= selected]
        if not contacts or not region_valid(molecule, selected, units):
            continue
        if any(not any(neighbor.GetIdx() in selected for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors())
               for contact in contacts for atom in contact["ligand_atoms"]):
            continue
        boundary = []
        for bond in molecule.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (begin in selected) != (end in selected):
                inside, outside = (begin, end) if begin in selected else (end, begin)
                positions = molecule.GetConformer().GetPositions()
                boundary.append(dict(source_atom=inside, omitted_atom=outside, bond_order=1, conjugated=bond.GetIsConjugated(),
                                     exit_vector_A=(positions[outside] - positions[inside]).tolist()))
        identity = dict(parent=packet["identifier"], atoms=list(atoms))
        result.append(dict(identifier="region:" + value_hash(identity)[:20], parent=packet["identifier"],
                           atom_indices=list(atoms), boundaries=boundary,
                           interaction_evidence=[contact["evidence_id"] for contact in contacts],
                           interaction_fingerprints=[contact["fingerprint"] for contact in contacts],
                           fragment_smiles=Chem.MolFragmentToSmiles(molecule, atomsToUse=list(atoms)),
                           heavy_atoms=len(atoms)))
    return result


def selected_regions(proposal, parent_lookup, region_lookup):
    selected = []
    for contribution in proposal["parent_regions"]:
        identifier = contribution["parent_identifier"]
        if identifier not in parent_lookup or not contribution["region_identifiers"]:
            raise ValueError("Unknown parent or empty region selection")
        regions = [region_lookup[region] for region in contribution["region_identifiers"]]
        if any(region["parent"] != identifier for region in regions):
            raise ValueError("Region belongs to a different parent")
        atoms = sorted({atom for region in regions for atom in region["atom_indices"]})
        parent = parent_lookup[identifier]
        molecule = Chem.MolFromMolBlock(parent["molblock"], removeHs=True)
        if len(atoms) >= molecule.GetNumAtoms():
            raise ValueError("Complete parents cannot be selected")
        selected.append(dict(parent=parent, molecule=molecule, atom_indices=atoms, regions=regions))
    if len(selected) != 2 or selected[0]["parent"]["identifier"] == selected[1]["parent"]["identifier"]:
        raise ValueError("Exactly two distinct parents are required")
    if topology(selected[0]["molecule"]) == topology(selected[1]["molecule"]):
        raise ValueError("Parent topologies must differ")
    return selected


def compile_condition(proposal, parents, regions, config):
    contributions = selected_regions(proposal, {parent["identifier"]: parent for parent in parents},
                                     {region["identifier"]: region for region in regions})
    assembled = Chem.RWMol()
    coordinates, sources, boundary_points = [], [], []
    for contribution in contributions:
        molecule, atoms = contribution["molecule"], contribution["atom_indices"]
        units = protected_units(molecule, config["repository"])
        if not region_valid(molecule, atoms, units):
            raise ValueError("Condition damages a protected chemical unit or has no legal boundary")
        mapping = {}
        for atom_index in atoms:
            mapping[atom_index] = assembled.AddAtom(Chem.Atom(molecule.GetAtomWithIdx(atom_index)))
            coordinates.append(molecule.GetConformer().GetAtomPosition(atom_index))
            sources.append(dict(parent_identifier=contribution["parent"]["identifier"], source_atom=atom_index))
        for bond in molecule.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin in mapping and end in mapping:
                assembled.AddBond(mapping[begin], mapping[end], bond.GetBondType())
        boundaries = sorted({boundary["source_atom"] for region in contribution["regions"] for boundary in region["boundaries"]
                             if boundary["omitted_atom"] not in atoms})
        if not boundaries:
            raise ValueError("No remaining attachment boundary")
        boundary_points.append(numpy.array(molecule.GetConformer().GetPositions())[boundaries])
    fixed_molecule = assembled.GetMol()
    conformer = Chem.Conformer(len(coordinates))
    conformer.Set3D(True)
    for atom_index, position in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, position)
    fixed_molecule.AddConformer(conformer)
    Chem.SanitizeMol(fixed_molecule)
    Chem.AssignStereochemistryFrom3D(fixed_molecule)
    split = len(contributions[0]["atom_indices"])
    clash = clashes(fixed_molecule.GetConformer().GetPositions()[:split],
                    [atom.GetAtomicNum() for atom in list(fixed_molecule.GetAtoms())[:split]],
                    fixed_molecule.GetConformer().GetPositions()[split:],
                    [atom.GetAtomicNum() for atom in list(fixed_molecule.GetAtoms())[split:]], config["clash_ratio"])
    if clash["severe"]:
        raise ValueError("Selected parent regions have a severe mutual clash")
    boundary_distance = float(numpy.linalg.norm(boundary_points[0][:, None] - boundary_points[1][None], axis=-1).min())
    sizes = size_range([entry["molecule"].GetNumAtoms() for entry in contributions], len(sources),
                       boundary_distance, config["linker_slack_atoms"], config["max_ligand_atoms"])
    selected_contacts = []
    for contribution in contributions:
        evidence = {identifier for region in contribution["regions"] for identifier in region["interaction_evidence"]}
        selected_contacts.append(dict(parent_identifier=contribution["parent"]["identifier"],
            evidence_ids=sorted(evidence), fingerprints=[contact["fingerprint"]
            for contact in contribution["parent"]["profile"]["interactions"] if contact["evidence_id"] in evidence]))
    bundle = dict(protocol=config["protocol"], proposal=proposal, sources=sources,
                  parent_source_hashes={entry["parent"]["identifier"]: entry["parent"]["source_member_sha256"] for entry in contributions},
                  fixed_molblock=Chem.MolToMolBlock(fixed_molecule), graph=chemical_graph(fixed_molecule),
                  selected_contacts=selected_contacts, size_range=sizes, mutual_clash=clash,
                  final_inpaint=False, apply_guidance=False)
    bundle["condition_hash"] = value_hash(bundle)
    return bundle


def opportunities(parents, regions, config):
    result = []
    for left, right in itertools.combinations(parents, 2):
        feasible = []
        for left_region in [region for region in regions if region["parent"] == left["identifier"]]:
            for right_region in [region for region in regions if region["parent"] == right["identifier"]]:
                proposal = dict(parent_regions=[dict(parent_identifier=left["identifier"], region_identifiers=[left_region["identifier"]]),
                                                dict(parent_identifier=right["identifier"], region_identifiers=[right_region["identifier"]])])
                try:
                    condition = compile_condition(proposal, [left, right], [left_region, right_region], config)
                    feasible.append(dict(region_identifiers=[left_region["identifier"], right_region["identifier"]],
                                         retained_interactions=sum(len(contact["evidence_ids"]) for contact in condition["selected_contacts"]),
                                         fixed_atoms=len(condition["sources"]), size_range=condition["size_range"]))
                except (ValueError, Chem.rdchem.MolSanitizeException):
                    continue
        feasible.sort(key=lambda entry: (-entry["retained_interactions"], entry["fixed_atoms"], entry["region_identifiers"]))
        result.append(dict(parents=[left["identifier"], right["identifier"]], feasible_region_pairs=len(feasible),
                           examples=feasible[:12]))
    return result
