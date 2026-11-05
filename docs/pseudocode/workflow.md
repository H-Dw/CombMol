# Complete workflow pseudocode — combmol-inpaint-v1

Written before implementation. The module contracts and failure semantics in `modules.md` are authoritative for this first campaign.

## Frozen campaign

    Targets := 6pyy_H7S_A_402 and 3w54_RNB_A_504
    ParentLimit := eight different atom/bond topologies per target
    FusionLimit := sixteen different unordered parent pairs per target
    SharedSeeds := 42, 100045, 200048, 300051, 400054
    ReplicatesPerSeed := five
    AttemptsPerCondition := twenty-five
    BatchUpperBound := fifty
    CampaignAttemptUpperBound := eight hundred
    Generation := native FLOWR.root fragment-growing inpainting
    GradientGuidance := disabled
    FinalInpaint := disabled
    Sampling := one hundred linear Euler steps
    EndpointScoreTime := 0.9999
    SevereClashRatio := 0.75
    RetainedConditionRMSD := strictly below 0.5 Angstrom per parent in the protein frame

## Preparation and selection

    RuntimeReceipt := RuntimePrepare(FrozenConfiguration)
    For each TargetCase in the declared target order:
        Acquire ownership of TargetCase
        SourceCandidates, TargetProtein, ReferenceLigand := ArchiveIndex(TargetCase)
        AlignedCandidates := ProteinAlign(SourceCandidates, TargetProtein)
        For each SourceCandidate:
            ScoreReceipt := EndpointScore(SourceCandidate, TargetProtein)
            BasicQuality := CheckChemistryAndProteinClashes(SourceCandidate)
            If basic quality and scoring are usable:
                InteractionReceipt := InteractionProfile(SourceCandidate, TargetProtein)
            Otherwise:
                Preserve the explicit rejection reason
        ParentPackets := ChemScreen(ScoredAndProfiledCandidates, ParentLimit)
        If fewer than two parents are available:
            Finalize TargetCase as no feasible combination
            Continue to the next TargetCase
        Catalogue := RegionCatalog(ParentPackets)
        ProposedFusions := FusionSelect(ParentPackets, Catalogue)
        ValidatedFusions := AuditEveryProposal(ProposedFusions, FusionLimit)
        For each ValidatedFusion in the submitted order:
            Condition := ConditionCompile(ValidatedFusion)
            Persist Condition and all source references

## Generation and evaluation

            For each BaseSeed in SharedSeeds:
                ExistingSlots := ReconcileAttemptLedger(Condition, BaseSeed)
                NativeBatch := ConstructOnlyPendingInitialStates(Condition, BaseSeed, ExistingSlots)
                If no pending slots exist:
                    Continue without a generation call
                Persist initialization identities and mark these slots started
                Try:
                    Terminals := NativeInpaint(NativeBatch)
                    Persist every returned terminal and build outcome
                On sampling exception:
                    Preserve exception and mark unresolved started slots as spent failures
                    Stop this target attempt for diagnosis; do not replace its slots
                For each saved terminal not yet evaluated:
                    Evaluation := FusionEvaluate(Terminal, Condition, TargetProtein)
                    Persist Evaluation
            Write the per-condition candidate table and summary
        Write TargetReport with all attempt denominators and parent comparisons
    Verify campaign budgets, identities and completeness
    Write CampaignReport and terminal status

## Background launch

    Commit and push tested implementation to the official main branch
    Verify the pushed commit identity
    Deploy that exact commit and verify all source/checkpoint hashes
    Launch the campaign independently of this chat
    Record process identity, launch ticks, boot identity and log locations
    Confirm background startup once with a bounded read
    Register one hourly heartbeat for this chat
    End the active execution turn without waiting for the whole experiment

## Hourly monitoring and recovery

    Read frozen launch identity, status, stage exits, ledger and recent logs
    If healthy or unchanged:
        End this heartbeat quietly
    If a new recoverable failure exists:
        Preserve the failed attempt and its complete evidence
        Diagnose the error without changing frozen inputs or scientific gates
        For a source defect:
            Create a tested repair commit, push it and deploy a separate pinned checkout
        Register the repair and its parent attempt
        Restart only incomplete preparation/evaluation or pending generation slots
        Keep started slots spent; do not append schemes, rounds or replacement candidates
        Submit long work to the background and end this heartbeat
    If the same error exceeds two repair recoveries or needs protocol/budget changes:
        Notify the user and hold the affected branch
    If every target is terminal:
        Reconcile all artifacts and failures
        Finish the report and notify the user
        Pause this monitor

## Pre-code consistency receipt

The design is internally consistent when the module names referenced above are defined, the shared budgets agree, every generation path records started slots first, terminal scoring never substitutes predicted chemistry, both source parents remain traceable, and no repair consumes a spent slot. This check must be recorded before adding implementation modules.
