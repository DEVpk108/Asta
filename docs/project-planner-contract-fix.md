# Project planner contract correction

Apply after ASTA-concurrent-model-services.patch, not instead of it.

- Classify normalized text but preserve the original project goal, including case-sensitive code, paths and expected output.
- Supply an explicit write/readback/run example. File verification belongs to filesystem.read_file with filesystem.file_content, matching path and decoded contents exactly.
- A rejected readback/command-verification proposal may be regenerated once with the validator error. The validator is unchanged; no rejected plan is executed or repaired by guessed code. Other failures, including invalid JSON, are not retried by this contract correction.
- Structured coding decisions use ASTA_CODING_AGENT_TEMPERATURE (default 0.0), independently of ordinary coding generation settings.
- Timeout, cancellation or server-reported output-limit termination cannot return a partial executable agent proposal. Timeout alone is not diagnosed as GPU OOM.
- “Are you generating the code?” and similar narrow status questions use background runtime state, not a model's guess. Exact project task status remains available.

Acceptance: use the disposable workspace and the exact main.py / ASTA_OK request, review approval, verify file readback and actual stdout. Use on_demand until full-stack resource tuning is validated. This correction does not resolve dual-mode GPU contention, guarantee that every model follows the contract, or add larger application-generation workflows.
