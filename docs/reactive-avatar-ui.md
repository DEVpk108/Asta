# Reactive avatar interface

## Views and actual telemetry

The default homepage centers a procedural 3D humanoid particle avatar. Open chat switches to the conversation workspace; Home or Escape returns without creating a session, clearing messages or stopping tasks. Both views keep the shared composer and project activity controls. The latest-response preview contains actual assistant text; code stays available in chat. No simulated model thought stream or fabricated progress is shown.

The avatar follows foreground runtime modes (idle, listening, thinking, speaking, executing, approval, error). Speech-output audio level drives its pulse and brightness while speaking; stale samples decay after 650 ms. It is expressive visualization, not phoneme-based lip synchronization. Background coding remains separately reported in project activity. Approvals and tool contracts are unchanged from the workspace interface; navigation never approves actions.

## Render budget and accessibility

One shared renderer is moved between the main stage and compact chat slot; no second WebGL context is created for chat. It reuses the existing particle geometry with a 12,000-particle cap, device-pixel-ratio cap of 1.25 and render-buffer dimension cap of 768 px. The low-power context hint is advisory and does not select the laptop iGPU. Low-power rendering avoids floating-point HDR targets.

Scheduling caps are 30 fps for active homepage states, 12 fps for homepage idle or coding loading/generation, 15 fps for chat speaking and 6 fps for other chat states. These are ceilings, not measured frame-rate guarantees. Hidden windows stop scheduling. Disconnected, manually paused or reduced-motion states render statically. Page teardown releases renderer resources. Context loss or unavailable WebGL switches to a labeled, audio-reactive 2D fallback; the text/approval interface remains usable even if both renderers fail.

Reduced motion and an explicit pause control are supported. The pause control is available on the homepage; its choice carries into chat. State remains readable without motion. No mandatory animation, rapid flashing or autoplay video is used. Desktop and narrow layouts retain task/approval controls; narrow layouts place project activity below conversation/home content.

## Validation and Windows acceptance

Run `node tests/js/test_reactive_avatar.cjs` for dependency-free controller checks. Run the Python planner, HUD, workspace, sandbox and model-router regression tests. Local Chromium checks use simulated IPC telemetry to test both views, audio state, fallback, reduced motion, safe message rendering, retained history, approvals, connection loss and narrow layouts. They do not establish Windows GPU memory or voice-stack performance.

Keep `ASTA_MODEL_MODE=on_demand` for initial hardware acceptance. Use a fresh disposable workspace and request creation of `main.py` containing only `print("ASTA_OK")`, then read it back and run it in the sandbox. Review and explicitly approve the exact command. Require actual stdout `ASTA_OK` and return code 0; planning, code display or button acknowledgment alone is not success. Monitor GPU usage with the full voice stack active before considering dual mode. Pause visual motion if additional headroom is required.
