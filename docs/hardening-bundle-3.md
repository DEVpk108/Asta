# Hardening Bundle 3 — operational, training, and repository cleanup

Base: `fix/hardening-bundle-2`, which passed 1225 tests on the Windows development
machine. This is the third bundled patch, not a claim that the whole product
roadmap is implemented. See `review-fix-status.md` for explicit remaining scope.

## Runtime changes

- AI, voice, speech, memory and tool-runtime diagnostic prints use standard
  module loggers (105 call sites). `main()` configures a console handler; log
  level is configurable, JSON formatting is optional, and rotating file logs are
  opt-in (5 MiB, three backups). Host/library handlers are preserved. This is not
  a global print monkey patch; other modules and CLI progress still use prints.
  Do not enable file logging for private conversations without considering data
  retention. Python logging configuration is not telemetry or trajectory tracing.
- Quick arithmetic uses a bounded allowlist AST evaluator instead of `eval()`.
  Spoken negatives, signed factors and negative cube roots work. Unsupported
  syntax, complex/nonfinite/oversized values and exponent chains are rejected.
- Windows process-tree termination uses `taskkill /PID <pid> /T` as an argument
  vector, with a timeout and no shell interpolation. POSIX SIGTERM is unchanged.
  Authority checks remain in the dispatcher, not bypassed by the tool.
- Application process queries have a short 0.5-second cache and a linear-time
  name-indexed dedupe pass. **Post-action verification and close-app rescans
  explicitly invalidate it** so stale pre-action state cannot certify execution.
  No hardware latency improvement is claimed without Windows measurement.
- Workspace persistence skips writes when its sanitized persisted fields are
  unchanged; environment/hardware-only updates no longer rewrite the JSON file.
  Failed saves remain retryable. This is not a general event-bus rewrite.
- UI search fallback pastes literal clipboard text instead of treating user
  punctuation as SendKeys commands. As with the earlier clipboard safety fix,
  it does not immediately restore an old clipboard payload. PowerShell process
  startup consolidation remains a separate performance improvement.

## Wakeword training

`ai/wakeword/audio_contract.py` defines **16000 Hz, mono, 16-bit PCM** for prepared
training WAVs. Positive/negative validators, augmentation, dataset combination,
conversion and feature preparation use the same rate constant. Native inference
already uses 16 kHz. Previously feature extraction resampled some inputs, so
mismatched intermediate rates were not proof that every trained model was wrong.

Piper output is genuinely resampled through FFmpeg before acceptance, not merely
header relabeled. Native-rate WAVs need no conversion. Conversion uses a temporary
file, validation and atomic replacement; failed conversions retain the input.
FFmpeg is a separate offline-training requirement. Stage 01 uses configured
canonical wake phrases rather than the unconfigured `Asta` literal. Resumed
positive generation rejects old-rate files with an actionable message rather
than silently mixing rates. Manifest membership is loaded once and updated in
memory. The production English preparer's repository-root calculation is fixed.

The two numbered stage-05 entry points now lead to the production 3000-sample
preparer. The duplicate training entry point delegates to the canonical trainer.
Feature extraction uses installed openwakeword, without forcing a vendored source
path into `sys.path`. The old AI wakeword module is a deprecated forwarding import
to the active voice implementation; the live detector thresholds are unchanged.

**Existing datasets/manifests are not automatically modified.** Back up them and
any trained models first. For a new training run, generate clean 16 kHz data in a
separate location or explicitly regenerate the affected stages, then regenerate
manifests/checksums, revalidate and retrain. Do not change sample-rate labels or
manifest checksums without regenerating the actual corresponding data. This patch
has not retrained or benchmarked a wakeword model's accuracy/false activations.

## Packaging and unused code

- `pyproject.toml` centralizes runtime requirements and optional extras. Existing
  requirements files forward to the same specification. Use commands from the
  repository root. Full runtime: `python -m pip install -e ".[kokoro,dev]"`.
  Optional training: `python -m pip install -e ".[wakeword-training]"` plus FFmpeg.
  Preserve/install the desired CUDA PyTorch build separately when appropriate.
  The spec is **not a reproducible release lockfile**, and the Python wheel is
  **not** a standalone Electron/model distribution. Existing environments do not
  need reinstalling just to apply this patch and run existing tests.
- Removed unused Moonshine/Polly implementations and duplicate screenshot opener
  files; the manual voice harness uses the canonical recognizer and starts only
  under `main()`. The runtime's registered screenshot tool is unchanged.
- Face recognition moved out of the runtime package to opt-in
  `experiments/face_identity.py`. It never unpickles embeddings, accepts a finite
  numeric `(N,128)` `.npy` array, bounds its frame loop and releases the camera on
  success, failure or interruption. It is **not an authorization mechanism**.
  Enrollment/migration is intentionally not automatic; do not unpickle untrusted
  artifacts. Optional OpenCV/face-recognition dependencies are not runtime deps.
- Eight original Word documents now have readable Markdown counterparts in
  `docs/reference/`, including tables. Originals are retained as source records.

## Data preservation and remaining repository decisions

The patch does **not** delete recordings, original documents, the tracked
biometric pickle, or the vendored openwakeword tree. New ignore rules prevent
future accidental additions, but **ignoring a tracked file does not untrack it or
erase it from Git history**. The read-only inventory command is:

```powershell
python scripts/audit_repository.py
```

The tracked biometric artifact remains a privacy concern before any public
release; removing its unsafe runtime loader does not remove the published data.
After a verified backup outside the repository, an owner can explicitly untrack
local recordings/biometrics with `git rm --cached` (use `-r` for a directory),
review the diff and commit. That command keeps the current local files. Do not
use plain `git rm` if the local copy must survive. Historical removal requires a
separately authorized, coordinated history rewrite; no force-push is part of this
patch. The vendored tree is no longer forcibly preferred by feature extraction,
but removal/replacement should follow installed-version/training compatibility
validation and license review, not blind deletion.

## Validation boundaries

Tests use synthetic WAVs/fakes, including one real FFmpeg duration-preservation
check when FFmpeg is installed. No actual microphone, camera, live Spotify login,
GPU model, Electron window or PowerShell execution was exercised on the review
machine. Full Windows regression and brief live smoke checks are still required.

### Review-machine results

- New regression file: 61 passed, repeated three consecutive times.
- Combined focused suite: 571 passed; existing quick-math suite: 8 passed.
- JavaScript: 11 test-runner entries passed, 0 failed.
- Full Linux: 1139 passed, 59 failed, 10 collection errors. Failure/error IDs
  exactly match the Bundle 2 baseline (1078 passed, 59 failed, 10 errors).
- A no-dependency wheel build and editable install succeeded in the isolated
  review environment. Imports resolved outside the checkout; the wheel was
  checked not to include recordings, biometric artifacts or vendored sources.
  This does not validate a clean Windows/CUDA dependency install.
- AST equivalence checks verified the five logging migrations changed logging
  calls, not their surrounding control flow.
