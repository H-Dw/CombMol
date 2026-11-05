---
name: parent-candidate-screening
description: Review measured unguided ligand candidates for chemically credible interaction-fragment combination. Use with a CombMol CandidatePacket batch and return evidence-bound keep or reject decisions.
metadata:
  version: "0.1.0"
---

# Parent Candidate Screening

Review every supplied candidate. The program establishes graph identity, validity, connectivity, finite measurements and severe clash gates. Do not recompute these facts in prose or change measured pIC50.

Assess whether the supplied chemical graph provides a credible source of transferable interaction regions. Consider formal charge, valence, aromaticity, conjugation, functional group integrity, obvious instability or reactivity, and the chemical roles supported by PLIP. Distinguish a concrete chemical liability from an uncertain preference; unusual chemistry alone is not evidence of failure.

Use the original graph and measured contacts. An interaction label does not prove experimental binding. PoseBusters is diagnostic in this campaign and is not an additional all-checks admission gate. Do not impose an unstated affinity, molecular-weight, SAScore or similarity threshold.

Return exactly one decision per supplied candidate in the JSON contract. Cite its supplied source evidence and give a short chemical rationale. Return decisions only; do not invent structures, atom indices, scores, coordinates or evidence. Ranking and graph deduplication are deterministic downstream operations.

For input/output details read [the contract](references/contract.md). The caller supplies the current JSON Schema, which is authoritative.
