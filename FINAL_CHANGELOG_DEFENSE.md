# Final Changelog — Defense Closure

Audit date: 2026-09-12

No production architecture, model, dataset, sensor behavior, or experiment result was
changed. Historical evidence was preserved.

| Modification | Reason | Affected files |
|---|---|---|
| Added Architecture Revision R2 and the exact completed contribution list | Record the final Phase 11 scope freeze and align the source of truth with the completed report | `docs/PROJECT_ARCHITECTURE.md`; outer `Instruction/PROJECT_ARCHITECTURE.md` |
| Marked Phases 12–14 and EWC/OTA as future work | Prevent future work from being presented as missing implementation or completed contribution | Both `PROJECT_ARCHITECTURE.md` copies |
| Removed continual-learning/OTA from the production diagram and defense demo sequence | Keep the production/demo story within frozen scope | Both `PROJECT_ARCHITECTURE.md` copies |
| Synchronized the outer architecture copy with the repository's `pose-v1` rules | Eliminate competing canonical copies while preserving the completed dashboard implementation | outer `Instruction/PROJECT_ARCHITECTURE.md` |
| Replaced obsolete Phase 11 “in progress” status | Reflect the canonical completed M11 checkpoint | `README.md`; `docs/phase11_database_dashboard.md` |
| Closed Phase 11 dashboard/hardware checklist items already supported by the document's recorded validation | Remove internal contradiction between measured completion narrative and unchecked boxes | `docs/phase11_database_dashboard.md` |
| Corrected the MQTT protocol's obsolete “no failover” statement | Match completed Phase 10 behavior: effective LOCAL, logged transition, cached-result reuse, no duplicate inference | `docs/mqtt_protocol.md` |
| Added the final audit, limitations, demo checklist, claims checklist, and traffic-light state | Provide the required defense handoff | `DEFENSE_READINESS_REPORT.md` |
| Added this complete modification record | Provide traceability for defense-only changes | `FINAL_CHANGELOG_DEFENSE.md` |
| Added final live system and hardware verification record | Freeze the actual software, stack, firmware, telemetry, dashboard, and failover results used for packaging | `FINAL_DEFENSE_VERIFICATION.md` |

## Verification record

- `pytest -q`: 343 passed; 7 third-party deprecation warnings.
- `npm run build`: passed; Vite emitted a non-blocking bundle-size warning.
- `docker compose config --quiet`: passed.
- PlatformIO was located at the standard user installation path. The production firmware
  build and COM10 upload passed; the coordinated on-device failover suite passed all six
  Unity cases. See `FINAL_DEFENSE_VERIFICATION.md` for the wrapper compatibility note.

## Deliberately unchanged artifacts

- `ProjectDocument.docx`: already states R2 scope freeze, Phase 0–11 completion, correct
  LOCAL/CLOUD and failover behavior, metric provenance, and future-work boundaries.
- `ProjectDefinitionAlirezaNobakht.pdf`: retained as the original project-definition
  artifact; later evidence-driven architecture revisions are documented rather than
  silently rewriting the proposal.
- Phase 7–9 four-action datasets, split-controller research code, manifests, logs, and
  closure notes: retained as immutable historical/experimental evidence, not production.
- `ml/continual/` and `config/continual_learning_v1.json`: retained as non-production
  future-work scaffolding; no Phase 12 execution or claims were added.
- Model files and measured experiment reports: unchanged to protect reproducibility.
