import math
import numpy
from rdkit import Chem


def topology(molecule):
    copy = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(copy)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def protein_atoms(path):
    coordinates, elements = [], []
    for line in open(path, encoding="utf-8"):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        symbol = line[76:78].strip().capitalize()
        if not symbol:
            symbol = line[12:16].strip().lstrip("0123456789")[0].upper()
        if symbol in ("H", "D"):
            continue
        elements.append(Chem.GetPeriodicTable().GetAtomicNumber(symbol))
        coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coordinates:
        raise ValueError("Empty protein heavy-atom coordinates")
    return numpy.array(coordinates), elements


def clashes(left_coordinates, left_elements, right_coordinates, right_elements, threshold=0.75):
    left_coordinates = numpy.asarray(left_coordinates, dtype=float)
    right_coordinates = numpy.asarray(right_coordinates, dtype=float)
    if not numpy.isfinite(left_coordinates).all() or not numpy.isfinite(right_coordinates).all():
        raise ValueError("Nonfinite clash coordinates")
    radii_left = numpy.array([Chem.GetPeriodicTable().GetRvdw(int(element)) for element in left_elements])
    radii_right = numpy.array([Chem.GetPeriodicTable().GetRvdw(int(element)) for element in right_elements])
    distances = numpy.linalg.norm(left_coordinates[:, None] - right_coordinates[None], axis=-1)
    ratios = distances / (radii_left[:, None] + radii_right[None])
    indices = numpy.argwhere(ratios < threshold)
    return dict(severe=bool(len(indices)), count=int(len(indices)),
                min_distance_A=float(distances.min()), min_ratio=float(ratios.min()),
                pairs=[dict(left_atom=int(row[0]), right_atom=int(row[1]),
                            distance_A=float(distances[tuple(row)]), ratio=float(ratios[tuple(row)]))
                       for row in indices[:40]])


def proper_alignment(moving, reference):
    moving, reference = numpy.asarray(moving, float), numpy.asarray(reference, float)
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Alignment requires corresponding three-dimensional points")
    if not numpy.isfinite(moving).all() or not numpy.isfinite(reference).all():
        raise ValueError("Nonfinite alignment")
    moving_center, reference_center = moving.mean(0), reference.mean(0)
    left, _, right = numpy.linalg.svd((moving - moving_center).T @ (reference - reference_center))
    correction = numpy.eye(3)
    correction[-1, -1] = numpy.linalg.det(left @ right)
    rotation = left @ correction @ right
    translation = reference_center - moving_center @ rotation
    return rotation, translation


def frame_rmsd(observed, reference):
    difference = numpy.asarray(observed, float) - numpy.asarray(reference, float)
    if not numpy.isfinite(difference).all():
        raise ValueError("Nonfinite condition RMSD")
    return float(numpy.sqrt(numpy.mean(numpy.sum(difference * difference, axis=-1))))


def size_range(parent_sizes, fixed_count, boundary_distance, slack=2, maximum=250):
    minimum_free = max(1, math.ceil(boundary_distance / 1.6) - 1)
    reference = max(max(parent_sizes), fixed_count + minimum_free + slack)
    lower = max(fixed_count + minimum_free, math.ceil(reference * 0.9))
    upper = min(maximum, math.floor(reference * 1.1))
    if lower > upper:
        raise ValueError("Condition exceeds native atom capacity")
    return dict(reference=reference, lower=lower, upper=upper, minimum_free=minimum_free,
                method="parent-reference-and-distance-lower-bound-v1", boundary_distance_A=boundary_distance)
