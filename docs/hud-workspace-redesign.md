# ASTA workspace interface

Apply after the concurrent-model-services patch and project-planner-contract fix.

## Design

Use a voice-first homepage with a reactive humanoid particle avatar and a separate conversation workspace with a compact avatar. Both views share history, messages, composer and the real project activity/approval panel. Use system typography, restrained accents, semantic status colors, light/dark conversation themes and immediate press feedback. Honor reduced motion, transparency and contrast preferences. The homepage keeps a dark stage for particle contrast. The existing WebGL renderer is reused with explicit render budgets; no GPU-memory improvement is claimed without measurement. See [reactive avatar behavior](reactive-avatar-ui.md).

The interface shows actual model states (including unloaded on-demand profiles), background phases and registered tool-step states. It does not display hidden reasoning or fabricate progress percentages, performance metrics, file downloads or success evidence. File read verification is labeled only when the runtime supplies verification metadata. Project status/cancel buttons use existing coordinator commands. Cancellation does not undo edits or forcibly interrupt a running tool.

## Approval safety

A real pending sandbox command includes its request ID, target, arguments, working directory and expiry. Approval/rejection is sent over the existing authenticated local transport. Python checks request existence, decision type and expiry; the original tool runtime still owns approval consumption, task ownership, permissions and execution. Buttons disable after a response is sent and at local expiry; backend expiry is authoritative. Arbitrary tool payloads are not exposed in this panel. Tools without a safe command preview can be rejected here but cannot be approved through the new buttons; their existing confirmation flow remains available.

No file contents, model rationale, full workspace paths or arbitrary task metadata enter project-step telemetry. Command previews are the intentional exception: the exact pending sandbox argv is needed for informed approval. Model messages are rendered with DOM text nodes and a small Markdown subset; model-generated HTML/links are never executed. Code copying reports success only when the clipboard operation succeeds.

Connection lifecycle controls whether commands are enabled. On socket loss Electron publishes a disconnected state. History, new conversation, session selection/deletion, voice conversation mode and native window controls remain connected to the existing IPC API. Deletion retains explicit confirmation. Focus mode preserves the project/approval panel instead of hiding safety controls.

## Validation

Run tests/test_hud_workspace_redesign.py and tests/test_hud_module.py, plus the planner/router/workspace/sandbox regression set. Local Chromium interaction checks cover history, voice toggles, command submission, safe code rendering, request-bound approval, disconnection, task failure and 390px layout. Inspect empty, task, approval, dark, narrow and disconnected states. These checks use simulated telemetry, not a running Windows Electron/voice stack. Final acceptance on Windows still requires the small ASTA_OK task, exact file readback, explicit execution approval and actual stdout.
