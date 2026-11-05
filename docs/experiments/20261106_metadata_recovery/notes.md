# PoseBusters source-installation metadata recovery

The first background attempt stopped for both targets before LLM selection or scientific generation. All 98 built 6pyy and 97 built 3w54 source molecules had endpoint scores and saved PLIP profiles, but diagnostic version collection raised `PackageNotFoundError` for PoseBusters. The module itself was importable; successful import did not imply an installed distribution metadata directory.

The repair first queries distribution metadata, then an importable module's declared `__version__`. If neither provides a version, the receipt records null. Missing version metadata does not fabricate a diagnostic result or bypass PoseBusters: its actual enabled checks still run, and diagnostic exceptions remain explicit.

Two regression cases cover a source installation with real diagnostic calls and an explicitly unknown version. Old failure receipts are retained in `runs/attempt_001` and the compressed pre-recovery archive named in `recovery.json`. The science budget spent before recovery is zero. The next independent checkout resumes the same frozen source scores, scientific configuration and attempt ledger under `attempt_002`.
