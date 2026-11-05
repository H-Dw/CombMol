# Screening contract

Input contains `candidates`, each with an immutable identifier, source evidence identifier, measured pIC50, full atom/bond graph, protein-frame coordinates, PLIP interaction records and basic quality measurements.

Output example:

```json
{"decisions":[{"candidate_identifier":"target/source-node","keep":true,"evidence_ids":["source:provided-hash"],"rationale":"The intact amide and aromatic region provide credible contact-bearing fragments; the supplied formal charge is chemically consistent."}]}
```

This example illustrates shape only. Use actual supplied identifiers. Account for every input exactly once, including rejected candidates. A short decision summary is sufficient; no private reasoning transcript is requested.
