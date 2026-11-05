# CombMol

Combine complementary, experimentally grounded interaction regions selected by a language model, then generate a connected ligand using native FLOWR.root fragment inpainting. No gradient guidance is used.

Implementation starts with the [module pseudocode](docs/pseudocode/modules.md) and [workflow pseudocode](docs/pseudocode/workflow.md). These documents precede implementation code and define the source identity, budget, failure and recovery rules.

The first campaign uses the archived unguided populations for `6pyy_H7S_A_402` and `3w54_RNB_A_504`, at most eight graph-distinct parents and sixteen parent pairs per target, and twenty-five attempts per accepted condition. Failed scientific attempts are retained, not replaced.

Model weights, datasets, credentials and generated results do not belong in this repository.
