# ASTA HUD — desktop application

A recreation of the holographic AI-agent interface from the reference video, rebuilt as a desktop app
(Electron + a hand-written WebGL2 particle engine). No CDN, no runtime downloads — the whole visual stack
ships inside the app and runs offline.

## Run it

```bash
cd asta-desktop
npm install      # downloads Electron once
npm start
```

## Package an installer (optional)

```bash
npm run dist         # current platform
npm run dist:mac     # .dmg
npm run dist:win     # .exe (NSIS)
npm run dist:linux   # AppImage
```

## What's inside

| File | Role |
| --- | --- |
| `main.js` | Electron main process: frameless window, native menu, IPC window controls, autoplay unlock, single-instance lock |
| `preload.js` | Secure `contextBridge` API (`window.asta`) — no Node in the renderer |
| `renderer/engine.js` | Humanoid cross-section model + WebGL2 particle system + custom bloom pipeline |
| `renderer/fallback2d.js` | Canvas2D renderer used only if WebGL2 is unavailable |
| `renderer/app.js` | State machine, assembly sequence, HUD, Web Audio drone, voice cue, telemetry |
| `renderer/index.html`, `renderer/styles.css` | Desktop shell: titlebar, stage, HUD overlay, status bar |

## The figure

The centerpiece is built from an anatomical table of horizontal slices (`SECTIONS` in `engine.js`), each
with a half-width, half-depth and a `drop` value:

- **Head** — ovoid cranium widest at the temples, tapering through cheekbones to a narrow jaw and chin,
  with separate particle clusters forming the **ears**.
- **Neck** — a true cylinder that flares into the trapezius.
- **Shoulders** — wide deltoid span whose contour bands sag toward their outer edges, which is what
  produces the downward-sweeping arcs across the shoulders and the nested arcs over the chest.
- **Face core** — wavy amber bands inside the skull plus a hot center, so the glow reads through the
  front contour lines; shell particles crossing the face oval are tinted amber.
- **Neural filaments** — an amber trunk through the neck that branches into clavicle and inner paths,
  with a pulse of energy travelling along each filament.
- **Silhouette** — particles near the left/right edge of every slice are brighter and larger, giving the
  crisp white-cyan outline around the head and shoulders.
- **Dust** — loose particles drift upward around the body and above the crown.

The figure sways gently instead of spinning, so it stays face-on to the viewer.

## Interface

- **Boot sequence** — ~24,000 particles stream out of the source orb below the frame and assemble into
  the figure while the HUD counts `ASSEMBLING 0 → 100%`. On completion the readout switches to a pulsing
  `STATUS: SPEAKING` and a synthetic voice says “Ready”.
- **States** — IDLE / LISTEN / THINK / SPEAK change particle energy, ripple amplitude, warm-light mix,
  bloom strength and the waveform.
- **Colors** — electric cyan `#00E5FF` on the contours, warm amber `#FF9100` in the face and along the
  neural filaments.
- **Status bar** — live renderer info, particle count, FPS and uptime.

## Shortcuts

| Key | Action |
| --- | --- |
| `1` `2` `3` `4` | Idle / Listening / Thinking / Speaking |
| `R` | Re-assemble from the orb |
| `F` | Toggle fullscreen |
| `M` | Mute / unmute the ambient drone |

Menu equivalents live under **Core** and **View** (`Cmd/Ctrl+1..4`, `Cmd/Ctrl+R`, `Cmd/Ctrl+M`).

## Notes

- macOS uses the native traffic lights (inset title bar); Windows and Linux get the custom in-app window
  controls in the top-right.
- The renderer also works if you simply open `renderer/index.html` in a browser — window controls are
  hidden and audio waits for the first click.
