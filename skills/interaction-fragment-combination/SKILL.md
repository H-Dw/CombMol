---
name: interaction-fragment-combination
description: Select complementary contact-bearing regions from two measured parent ligands for native FLOWR inpainting. Use with a CombMol region catalogue and compatibility evidence; select regions without designing a complete fused molecule.
metadata:
  version: "0.1.0"
---

# Interaction Fragment Combination

Choose at most sixteen promising combinations from the supplied parents. Use two different parent graphs per proposal and at most one proposal for each unordered parent pair. List proposals in priority order. Return fewer when evidence does not support more.

## Evidence and chemistry

Read parent atom/bond graphs, formal charges, aromaticity, functional groups, source poses and real PLIP contacts. Select only registered regions and supplied evidence. Preserve complete rings and functional groups. Retain the donor, acceptor, aromatic or charged chemical environment needed for each intended contact.

Treat the common protein frame as fixed. Whole-parent overlap is not disqualifying: compare the regions actually selected. Consult the measured compatibility examples and catalogue boundaries. Do not move, rotate, merge or relabel atoms to evade a clash.

## Complementary-region decisions

Consider which supported contacts each parent contributes, whether the contributions are complementary rather than redundant, and whether the remaining attachment sites and exit vectors leave plausible space for a generated connection. Evaluate increases in size and flexibility as uncertainty, not as guaranteed affinity improvement.

Choose contact-bearing regions, their necessary internal bonds and complete functional groups. Never select both entire molecules, specify a new connecting bond or construct the final molecular graph. Native inpainting generates free atoms and their connections. Condition atoms from different parents have no preassigned bond, so a direct bond between fixed parent atoms is not an available shortcut.

## Submission

Return the supplied JSON contract. Each proposal names two parents and their registered region identifiers, cites relevant source and interaction evidence, and briefly states interaction complementarity, chemical integrity, attachment feasibility and uncertainty. These are concise decision summaries, not a detailed internal reasoning transcript.

Account for unselected parent pairs with a short reason, including insufficient evidence or the sixteen-proposal budget. A zero-proposal response is valid if no combination is defensible. Do not invent measurements or claim experimentally validated binding.

For serialization details read [the contract](references/contract.md). The caller supplies the authoritative schema and deterministic audit feedback. Correct an invalid submission using that feedback without changing source data or the protocol.
