# Module pseudocode — protocol combmol-inpaint-v1

This specification is written before implementation. Names retain their keywords; do not introduce one-letter or two-letter shorthand. External field names such as `node_id` remain unchanged.

## Shared contracts

- `CandidatePacket`: target identity, archive/member hashes, exact source `node_id`, original atom ordering, graph, pose, endpoint score, interaction evidence, protected chemical units and quality measurements.
- `ScreenDecision`: candidate identifier, keep/reject, source evidence identifiers, short chemical rationale. No coordinate, graph or score edits.
- `FusionProposal`: exactly two graph-distinct parent identifiers; catalogue region identifiers for each parent; priority order; short interaction-complementarity, chemical-integrity, attachment-feasibility and uncertainty summaries. No invented connecting graph.
- `ConditionBundle`: selected source atoms and their provenance, original coordinates transformed only with the protein, atom features and within-fragment bonds, fixed mask, permitted size range and frozen source hashes. Different parents have no preassigned connecting bonds.
- `AttemptRecord`: condition identity, base seed, replicate slot, lifecycle, source revision, initialization hash, actual terminal artifacts, score, diagnostics and error. A started attempt cannot be reset to pending.
- Every cached artifact binds its input identity and protocol. Missing, stale or ambiguous records are never accepted as valid measurements.

## RuntimePrepare

Input: frozen configuration and source checkout.

    Check vendored FLOWR source and checkpoint hashes.
    Check runtime imports, accelerator access and configured model route.
    Freeze configuration, software versions and input identities.
    Reject gradient guidance or terminal condition overwriting.
    Output runtime receipt; stop before sampling on failure.

## ArchiveIndex

Input: dataset root and exact target identifier.

    Resolve archive from the dataset summary.
    Read only members belonging to the exact target; reject unsafe paths.
    Read the campaign-wide records and all built molecule members.
    Match structures using their node_id property, never record position.
    Require unique identities and agreement with build-success records.
    Preserve original atom order, pose, chemistry and source hashes.
    Output indexed candidates and the unchanged target protein/reference ligand.

## EndpointScore

Input: actual molecule graph and coordinates, frozen protein context.

    Prepare native FLOWR features for the supplied structure.
    Verify coordinate frame, tensor dimensions, devices and integer masks.
    Score the actual discrete endpoint at time 0.9999 with zero coordinate noise.
    Never substitute a decoded prediction for the submitted molecule.
    Return finite pIC50 or an explicit scoring failure; preserve historical scores separately.

## InteractionProfile

Input: original molecule and target protein.

    Add only analysis hydrogens without moving heavy atoms.
    Serialize a uniquely identified ligand and preserve its authoritative SDF graph.
    Run real PLIP with an explicit serial-to-source-atom mapping.
    Verify heavy-atom coordinates and chemical graph after serialization.
    Return interaction types, roles, source atom sets and protein residue identities.
    Cache by pose, graph, protein and profiler identity; do not turn tool failures into zero interactions.

## ProteinAlign

Input: source protein and campaign protein.

    When the proteins are identical, retain identity transformation.
    Otherwise require a complete unambiguous protein atom mapping.
    Fit a proper rigid transformation using protein atoms only.
    Apply the same transformation to all ligand atoms.
    Never fit, translate or rotate two ligands independently.

## ChemScreen

Input: scored candidates with geometry and interaction evidence.

    Reject invalid, disconnected or nonfinite structures.
    Reject severe protein heavy-atom clashes and unavailable/empty real interactions.
    Provide eligible structures and measured properties to LLM chemical screening.
    Require evidence-bound keep/reject decisions covering the supplied candidates.
    Sort kept candidates by measured pIC50, with stable identifiers breaking ties.
    Keep the highest-ranked usable pose per atom/bond topology, then take at most eight.
    Preserve stereochemistry even though conformers/stereochemistry alone do not create a new topology.
    Record all rejection reasons; never lower gates to fill the quota.

## RegionCatalog

Input: an accepted parent and its PLIP evidence.

    Identify protected functional groups and complete ring systems using frozen chemistry rules.
    Starting from each real interaction atom set, close over protected units and required bonds.
    Enumerate proper connected source regions separated only at permitted non-ring single bonds.
    Require a real interaction and at least one chemically available boundary per region.
    Reject complete-parent regions and partial functional groups/rings.
    Register source atoms, internal bonds, cut bonds, exit vectors and preserved interactions.

## FusionSelect

Input: at most eight parents and their region catalogue.

    Present full parent chemistry and measured region compatibility to one LLM.
    Let the LLM select complementary regions from two different parent topologies.
    Require short decision summaries and evidence identifiers, not a complete fused graph.
    Audit identifiers, source coverage, functional groups, boundaries and selected-region clashes.
    Require a different unordered parent pair for each proposal.
    Allow one bounded correction of invalid output, then preserve invalid-output evidence.
    Keep at most sixteen legal proposals in the submitted priority order.
    If none are feasible, finish the target with an explicit no-feasible-combination outcome.

## ConditionCompile

Input: a validated proposal.

    Copy only the selected original atom features, internal bonds and protein-frame coordinates.
    Combine source regions as disconnected partial condition components, without connecting parents.
    Require an interaction from both parents and available connection boundaries.
    Estimate a lower bound on free atoms from boundary separation; add two atoms of slack.
    Reference size is at least the larger parent size and condition size plus the connection allowance.
    Sample around that reference within ten percent, respecting the lower bound and native size limit.
    Freeze the bundle and all source mappings; reject empty or inconsistent masks.

## NativeInpaint

Input: frozen condition, shared seed set, five replicate slots per seed.

    Reconcile existing completed and started slots before any generation.
    For each shared base seed, construct five independent native initial states.
    Reserve and mark slots started before invoking the sampler.
    Call native fragment-growing inpainting: 100 linear Euler steps, no gradient guidance.
    Keep batch size at most fifty and do not pad with additional candidates.
    Do not enable final_inpaint, relax coordinates, repair results outside the frozen native policy or refill failures.
    Persist actual terminal tensors, structures and build failures for every slot.
    Ambiguous started slots remain spent and cannot be automatically resampled.

## FusionEvaluate

Input: actual terminal structures and the frozen condition.

    Score actual discrete returned endpoints with the shared scorer.
    Check source-slot chemistry, connectivity, clashes and source-fragment pose retention.
    Measure each parent's condition RMSD in the protein frame without ligand realignment.
    Profile real terminal interactions and require at least one selected expected interaction from each parent.
    Record PoseBusters, SAScore, QEDScore, size, pIC50 and all unevaluable states.
    Label a successful fusion only when chemistry, both condition regions and real interactions survive.
    Report all-attempt denominators separately from valid-fusion score means and best scores.

## CampaignResume

Input: campaign receipts, attempt ledger and the pinned revision.

    Verify input identities and acquire exclusive campaign ownership.
    Reuse compatible completed stage receipts and terminal results.
    Re-evaluate saved terminals when only evaluation is incomplete.
    Continue pending slots; never reopen a started slot.
    Preserve previous attempts and revisions when a repair is deployed.
    Stop the affected branch if its scientific protocol would change.

## HourlyInspect

Input: launch identity, status, logs, receipts, budgets and version information.

    Perform one bounded inspection and validate process start ticks and boot identity.
    If healthy or unchanged, end this inspection without a message or polling loop.
    On failure, preserve evidence and classify dependency, transport, interface, code or scientific failure.
    Repair recoverable defects using a new tested commit and a new attempt directory.
    Restart only failed preparation/evaluation stages or still-pending generation slots.
    Limit the same error to two automatic repair recoveries; otherwise request user attention.
    Submit long tests and recovery work to the background, then end this inspection.
    On complete campaign termination, verify results, publish the report and pause the monitor.

## Consistency checks required before implementation

- Every stage consumes only declared, provenance-bound inputs and produces a declared output.
- Two parent contributions survive every transformation into a condition.
- No LLM response can alter measured scores, original coordinates or the scientific budget.
- All generation calls are preceded by durable started-slot records.
- Budget is at most sixteen times twenty-five per target, eight hundred across both targets.
- A quality failure is data; a tool failure is not a successful or zero-valued measurement.
- Empty conditions never cause silent de novo generation.
- Background execution does not rely on an active chat or a long polling loop.
