# Implementation validation

The pseudocode was committed first in `f0f58e5`, before implementation files existed. The first implementation is `d06d3ae`; `1726a72` corrects two required arguments in the pinned native parser. Neither engineering check generated scientific candidates.

At `1726a72`, the remote DTK Python 3.11 environment passed all 21 tests, including the actual native inpainting builder. Both target checkpoints loaded on the accelerator and scored two archived source molecules each. The JSON receipt preserves candidate identities and scores; these are engineering examples, not selected parents or scientific fusion results. The source system is uncentered, feature preparation restores the original protein frame, and the score operates on the actual discrete input molecule at endpoint time 0.9999.

The final launch also repeats the expanded CPU suite, verifies every vendored source byte and the checkpoint hash, and records its exact commit and process identity. The additional recovery test checks that saved terminal batches finish already-started slots without a second generation call or additional quota.

Local validation uses an isolated RDKit installation and skips tests requiring PyTorch. The two English skill entrypoints pass the skill format validator. These tests establish interface and accounting behavior; they do not establish affinity improvement or chemical synthesis feasibility.

Scientific choices remain explicit: PLIP supplies structure-based contacts, predicted pIC50 supplies ranking, and PoseBusters is diagnostic. Contact presence does not establish experimentally measured binding. The condition assembly retains original parent coordinates and internal chemistry without specifying a fused molecule. Native terminal drift is measured rather than overwritten. Linker size is a distance-based allocation heuristic.
