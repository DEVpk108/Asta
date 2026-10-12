"use strict";
// Dependency-free checks for the Core particle orb renderer.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function fakeCanvas(width, height) {
  const calls = { arc: 0, rect: 0, fill: 0, clear: 0, styles: new Set() };
  const gradient = { addColorStop() {} };
  const ctx = {
    set fillStyle(v) { calls.styles.add(String(v)); },
    get fillStyle() { return ""; },
    globalCompositeOperation: "source-over",
    setTransform() {},
    clearRect() { calls.clear++; },
    fillRect() {},
    beginPath() {},
    arc(x, y, r) {
      assert.ok(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(r), "finite geometry");
      calls.arc++;
    },
    fill() { calls.fill++; },
    moveTo() {},
    rect(x, y, w) {
      assert.ok(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(w), "finite geometry");
      calls.rect++;
    },
    createRadialGradient() { return gradient; },
  };
  return { canvas: { clientWidth: width, clientHeight: height, width: 0, height: 0, getContext: () => ctx }, calls };
}

const context = { window: { devicePixelRatio: 2 }, Math, Number, Object, Array, Error };
vm.runInNewContext(
  fs.readFileSync(path.resolve(__dirname, "../../hud/renderer/core_avatar.js"), "utf8"),
  context,
);
const Core = context.window.AstaCoreAvatar;
assert.ok(Core && Core.create, "renderer is exported");
for (const mode of ["idle", "listening", "thinking", "speaking", "executing", "approval", "error", "offline"])
  assert.ok(Core.MODES.includes(mode), `${mode} has a visual`);

// Budget caps: particle count and backing-store size.
const big = fakeCanvas(1600, 900);
const engine = Core.create(big.canvas, { maxParticles: 2400, maxDpr: 1.25, maxDimension: 768 });
assert.equal(engine.ok, true);
assert.equal(engine.count, 2400);
assert.ok(Math.max(big.canvas.width, big.canvas.height) <= 768, "render buffer capped");
assert.equal(Core.create(fakeCanvas(10, 10).canvas, { maxParticles: 99999 }).count, 4000);

// Every mode renders finite geometry; colours differ between modes.
const palettes = {};
let time = 0;
for (const mode of Core.MODES) {
  const probe = fakeCanvas(600, 400);
  const r = Core.create(probe.canvas, { maxParticles: 800 });
  for (let i = 0; i < 5; i++) r.render({ mode, time: (time += 0.05), audio: mode === "speaking" ? 0.8 : 0 });
  assert.ok(probe.calls.arc + probe.calls.rect >= 800, `${mode} draws particles`);
  palettes[mode] = [...probe.calls.styles].slice(0, 40).join("|");
}
assert.notEqual(palettes.idle, palettes.thinking, "thinking is visually distinct");
assert.notEqual(palettes.idle, palettes.approval, "approval is visually distinct");

// Still frames (paused / reduced motion) snap to the state instead of animating.
const still = fakeCanvas(600, 400);
const s = Core.create(still.canvas, { maxParticles: 400 });
s.render({ mode: "idle", time: 0, still: true });
s.render({ mode: "thinking", time: 0, still: true });
assert.ok([...still.calls.styles].some((c) => c.startsWith("rgba(205,175,255")), "paused frame shows the current state");

// Compact chat slot draws fewer particles; dispose releases the buffer.
const tiny = fakeCanvas(58, 58);
const t = Core.create(tiny.canvas, { maxParticles: 2400 });
t.render({ mode: "idle", time: 0.1 });
assert.ok(tiny.calls.arc + tiny.calls.rect < 1200, "compact avatar is lighter");
t.dispose();
assert.equal(tiny.canvas.width, 1);

console.log("Core avatar renderer checks passed: modes, budgets, still frames, compact view and cleanup.");
