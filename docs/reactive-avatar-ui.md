# Reactive avatar interface

## Views and actual telemetry

The default homepage centers the ASTA Core: a 3D-projected particle orb (`hud/renderer/core_avatar.js`). Open chat switches to the conversation workspace; Home or Escape returns without creating a session, clearing messages or stopping tasks. Both views keep the shared composer and project activity controls. The latest-response preview contains actual assistant text; code stays available in chat. No simulated model thought stream or fabricated progress is shown.

The avatar follows foreground runtime modes (idle, listening, thinking, speaking, executing, approval, error). Each mode has its own colour and motion: idle (blue, slow rotation), listening (cyan ripples), thinking (violet swirl with a warm centre), speaking (warm white), executing (green swirl), approval (amber pulse), error (red, subdued) and offline (grey, still). Mode changes blend smoothly. Speech-output audio level drives the speaking pulse and brightness; stale samples decay after 650 ms. It is expressive visualization, not phoneme-based lip synchronization. Background coding remains separately reported in project activity. Approvals and tool contracts are unchanged from the workspace interface; navigation never approves actions.

## Render budget and accessibility

One shared renderer is moved between the main stage and compact chat slot; no second WebGL context is created for chat. The Core is drawn with Canvas 2D (no WebGL context or GPU bloom pipeline) using a 2,400-particle cap, a device-pixel-ratio cap of 1.25 and a render-buffer dimension cap of 768 px. The compact chat avatar draws one third of the particles.

Frames are paced by `requestAnimationFrame` (display vsync), never by timers, because Windows timer granularity makes timer-paced animation stutter. The homepage draws every display frame; the compact chat avatar and the homepage during coding loading/generation draw every other frame. Particles are drawn in ten depth bands (one fill per band) to keep each frame cheap. Hidden windows stop scheduling. Disconnected, manually paused or reduced-motion states render statically. Page teardown releases renderer resources. If the Core renderer fails, the labeled 2D fallback is used; the text/approval interface remains usable even if both renderers fail.

Reduced motion and an explicit pause control are supported. The pause control is available on the homepage; its choice carries into chat. State remains readable without motion. No mandatory animation, rapid flashing or autoplay video is used. Desktop and narrow layouts retain task/approval controls; narrow layouts place project activity below conversation/home content.

## Themes, branding, and listening state

The home view follows the app theme through `--home-*` variables in `reactive_avatar.css` (light by default, dark via `data-theme="dark"` or the OS preference). Only the avatar stage stays dark so the particle core keeps its contrast. The ASTA atom mark is inline SVG (`.asta-logo`) in the title bar and chat welcome; `hud/assets/asta-logo.svg` and `hud/assets/asta-icon.png` are the standalone mark and window icon.

The HUD shows **Listening** only while a conversation is actually open. `VoiceModule` emits `conversation_state_changed(active=...)` when the wake-word follow-up window opens or closes, and `HUDModule.on_speech_finished` returns to idle when no conversation is active, so turning conversation mode off no longer leaves the avatar stuck on Listening.

## Validation and Windows acceptance

Run `node tests/js/test_reactive_avatar.cjs` and `node tests/js/test_core_avatar.cjs` for dependency-free controller and renderer checks. Run the Python planner, HUD, workspace, sandbox and model-router regression tests. Local Chromium checks use simulated IPC telemetry to test both views, audio state, fallback, reduced motion, safe message rendering, retained history, approvals, connection loss and narrow layouts. They do not establish Windows GPU memory or voice-stack performance.

Keep `ASTA_MODEL_MODE=on_demand` for initial hardware acceptance. Use a fresh disposable workspace and request creation of `main.py` containing only `print("ASTA_OK")`, then read it back and run it in the sandbox. Review and explicitly approve the exact command. Require actual stdout `ASTA_OK` and return code 0; planning, code display or button acknowledgment alone is not success. Monitor GPU usage with the full voice stack active before considering dual mode. Pause visual motion if additional headroom is required.
