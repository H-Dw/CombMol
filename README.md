# CombMol

Combine complementary, structure-grounded interaction regions selected by a language model, then generate a connected ligand using native FLOWR.root fragment inpainting. No gradient guidance is used.

Implementation starts with the [module pseudocode](docs/pseudocode/modules.md) and [workflow pseudocode](docs/pseudocode/workflow.md). These documents precede implementation code and define the source identity, budget, failure and recovery rules.

The first campaign uses the archived unguided populations for `6pyy_H7S_A_402` and `3w54_RNB_A_504`, at most eight graph-distinct parents and sixteen parent pairs per target, and twenty-five attempts per accepted condition. Failed scientific attempts are retained, not replaced.

Model weights, datasets, credentials and generated results do not belong in this repository.

## Run

The application uses the existing DTK environment on the authorized server. It does not install a different CUDA/PyTorch stack.

```bash
export PYTHONPATH="$PWD/src:$PWD/third_party/flowr_root"
python -m pytest tests -q
python -m combmol.cli preflight --config configs/campaign.json
bash scripts/run_campaign.sh configs/campaign.json attempt_001
python -m combmol.cli inspect --root /workspace/combmol_campaign_20261106
```

For an independent background process, use `python scripts/launch_background.py --config configs/campaign.json --attempt attempt_001` instead of the foreground run command. The launcher records the pinned commit, scientific plan, process start ticks and boot identity, and hides interactive-shell startup output. Use a new attempt identifier for a repair; it resumes the same frozen campaign and slot ledger.

Set `DEEPSEEK_API_KEY` in the process environment. The model is `deepseek-flash`; its API requests contain evidence packets and strict JSON contracts. Credentials are never configuration or receipt fields. The two English skill entrypoints are the single prompt sources; schemas live in `schemas/` and are included in each request.

The pinned native sampler uses 100 linear Euler steps, no gradient guidance, no additional optimization, no terminal condition overwrite, and no decode-time valence repair. Native categorical Euler guarding remains enabled. All effective arguments are saved with the target. Each seed produces five slots; the configured batch size of fifty is an upper bound, not padding or an instruction to generate additional molecules.

Graph uniqueness ignores atom order, conformation and stereochemistry-only changes while retaining the source stereochemical structure. Fragment enumeration protects native functional groups and complete rings, permits cuts only at unprotected non-ring single bonds and keeps a source bond adjacent to each retained interaction atom. Conjugated boundary bonds are flagged for the LLM rather than silently severing a protected functional group.

Selected fragments remain in their shared protein frame. The native condition fixes within-fragment bonds and no-bond relations across fixed fragments; free atoms must provide the new connection. The size policy uses a distance-based lower bound, two slack atoms and a ±10% reference-size range. This is a space-allocation heuristic, not a proof of synthetic or binding feasibility.

The attempt ledger is written before each five-slot sampling call. A started slot is never regenerated automatically, even if the process dies. Completed terminal structures can be re-evaluated at zero generation cost. Restart a failed run with a new `--attempt` identifier; old run receipts remain available. Do not change the scientific configuration or frozen selection to repair an engineering error.

Results include source indices, parent scores, PLIP and PoseBusters receipts, complete LLM request/decision records, condition provenance, per-slot tensors and SDFs, candidate CSVs, group summaries, the online protocol audit and a Chinese report. Missing measurements remain explicit. Only valid retained fusions contribute to the reported fusion mean/best pIC50.

## Upstream source

`third_party/UPSTREAM.json` binds every vendored file to canonical Git-object bytes from FLOWR.root commit `739a33c2aab74d074dd24605b9a2e34dc809ae9c`. The native source is unchanged. Its MIT license and the embedded RDKit functional-group attribution are retained. The user-owned EvoMolSteer PLIP serialization approach is adapted in `interactions.py` to preserve the authoritative SDF graph and original heavy coordinates.
