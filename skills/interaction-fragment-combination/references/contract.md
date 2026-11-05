# Combination contract

Input provides full parent graphs and source coordinates; real PLIP contacts; protected, registered source regions; and examples of geometrically compatible region pairs. Examples are opportunities, not measured fused molecules or predicted improvements.

Output is an object with `proposals` and `unselected_pairs`. Each proposal contains `identifier`, exactly two `parent_regions` entries, `evidence_ids`, and the four rationale strings named in the supplied schema. Parent entries contain `parent_identifier` and `region_identifiers`.

Evidence must include both source parents and at least one selected interaction from each parent. Atom indices are zero-based indices of the original stored source structure; select catalogue identifiers rather than inventing indices. The program compiles the partial condition graph. It does not infer a complete graph from your explanation.

The two source regions may contain several registered units. Their union must remain a proper subset of each parent with legal connection boundaries and no protected-unit damage. Different prose or identifiers do not make a repeated parent pair new.

After the final native prediction, no additional condition overwrite is applied. Drift and contact recovery are measured on the returned structure; the existence of a condition does not guarantee preservation.
