# Approval consent safety

Voice confirmation previously reduced any short reply beginning with `okay`,
`sure` or `yes` to `yes`. This lost refusals such as "okay don't do it" before
the AI approval handler could inspect them.

## Contract

- Voice preserves the complete reply, including negation and question marks.
- Voice and typed replies share `core/approval_language.py`.
- Ordinary approval accepts complete explicit phrases, not an affirmative word
  anywhere in a sentence. "yes please do it", "I confirm", "go ahead", "हाँ"
  and explicit "okay go ahead" still work.
- Bare "okay", "ok" and "sure", questions, conditionals, new commands, and
  mixed affirmative/negative replies never authorize execution.
- Unrecognized ordinary replies retain the existing behavior: cancel pending
  approval before handling the new user turn. Conservative parsing may require
  the user to restate consent with one of the explicit phrases.
- Background tasks still require the HUD's request-bound button or the exact
  `approve coding step` / `approve project step` command. Ordinary chat neither
  approves nor clears that background request.
- Request IDs, expiry, authority checks and single-use approval consumption are
  unchanged. The language helper does not itself grant tool authority.

## Validation

Run:

```sh
python -m pytest -q tests/test_approval_consent_safety.py tests/test_approval.py tests/test_authority.py tests/test_concurrent_models.py
```

The regression tests exercise the actual AI handler and approval/tool runtime
with an inert critical-risk tool. They also compile the actual voice adapter
method from source, avoiding the package's eager optional audio imports. This
checks the adapter and consent path, not live microphone recognition.

Before local Windows acceptance, use an inert test action: verify that "okay
don't do it", "okay stop", "yes don't", and "sure" never execute it; then verify
that a fresh request accepts "yes please do it" exactly once. Check the explicit
background approval phrase separately. No destructive commands are needed.