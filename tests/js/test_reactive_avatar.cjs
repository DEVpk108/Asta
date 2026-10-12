"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
function environment(failWebgl = false, reduced = false) {
  const ids = {},
    listeners = {},
    docListeners = {},
    rafs = new Map(),
    timers = new Map();
  let sequence = 0,
    now = 100,
    options,
    renders = 0,
    disposals = 0;
  function node(id) {
    if (ids[id]) return ids[id];
    const classes = new Set();
    return (ids[id] = {
      id,
      dataset: {},
      hidden: false,
      disabled: false,
      textContent: "",
      parentNode: null,
      classList: {
        toggle(name, value) {
          value ? classes.add(name) : classes.delete(name);
        },
        contains(name) {
          return classes.has(name);
        },
      },
      setAttribute(name, value) {
        this[name] = value;
      },
      addEventListener(name, fn) {
        this["on" + name] = fn;
      },
      appendChild(child) {
        child.parentNode = this;
      },
      focus() {
        this.focused = true;
      },
    });
  }
  const engine = {
    ok: true,
    count: 2400,
    render() {
      renders++;
    },
    dispose() {
      disposals++;
    },
  };
  const media = {
    matches: reduced,
    addEventListener(name, fn) {
      this.onchange = fn;
    },
  };
  const document = {
    hidden: false,
    getElementById: node,
    addEventListener(name, fn) {
      docListeners[name] = fn;
    },
  };
  const window = {
    asta: null,
    matchMedia: () => media,
    addEventListener(name, fn) {
      listeners[name] = fn;
    },
    dispatchEvent(event) {
      if (listeners[event.type]) listeners[event.type](event);
    },
    AstaCoreAvatar: {
      create(canvas, value) {
        options = value;
        if (failWebgl) throw new Error("no canvas");
        return engine;
      },
    },
    AstaFallback2D: {
      create() {
        return { ...engine, count: 0 };
      },
    },
  };
  const context = {
    window,
    document,
    performance: { now: () => now },
    Number,
    CustomEvent: function (type, value) {
      this.type = type;
      this.detail = value.detail;
    },
    requestAnimationFrame: (fn) => {
      const id = ++sequence;
      rafs.set(id, fn);
      return id;
    },
    cancelAnimationFrame: (id) => rafs.delete(id),
    setTimeout: (fn) => {
      const id = ++sequence;
      timers.set(id, fn);
      return id;
    },
    clearTimeout: (id) => timers.delete(id),
  };
  vm.runInNewContext(
    fs.readFileSync(
      path.resolve(__dirname, "../../hud/renderer/reactive_avatar.js"),
      "utf8",
    ),
    context,
  );
  const frame = () => {
    now += 100;
    const callbacks = [...rafs.values()];
    rafs.clear();
    callbacks.forEach((fn) => fn(now));
  };
  const tick = () => {
    const callbacks = [...timers.values()];
    timers.clear();
    callbacks.forEach((fn) => fn());
    frame();
  };
  return {
    window,
    document,
    node,
    frame,
    tick,
    listeners,
    docListeners,
    media,
    diagnostics: () => window.AstaAvatar.diagnostics(),
    stats: () => ({
      options,
      renders,
      disposals,
      raf: rafs.size,
      timers: timers.size,
    }),
  };
}
const env = environment();
env.frame();
assert.equal(env.window.AstaViews.current(), "home");
assert.equal(env.stats().options.maxParticles, 2400);
assert.equal(env.stats().options.maxDimension, 768);
assert.equal(env.stats().options.maxDpr, 1.25);
assert.equal(env.diagnostics().renderer, "core");
assert.equal(env.stats().raf + env.stats().timers, 0, "offline avatar must not loop");
assert.equal(env.diagnostics().lastFrame.mode, "offline");
assert.equal(env.diagnostics().lastFrame.still, true);
env.window.AstaAvatar.setReady(true);
env.window.AstaAvatar.setState({ mode: "speaking" });
env.window.AstaAvatar.setAudioLevel(1);
env.frame();
env.tick();
env.tick();
const loud = env.diagnostics().lastFrame.amp;
assert.equal(env.diagnostics().lastFrame.mode, "speaking");
assert.ok(env.diagnostics().lastFrame.audio > 0, "speech audio reaches the renderer");
const loudAudio = env.diagnostics().lastFrame.audio;
assert.equal(env.diagnostics().frameEvery, 1, "homepage animates every display frame");
assert.equal(env.stats().timers, 0, "animation is never timer-paced");
env.window.AstaAvatar.setAudioLevel(0);
for (let i = 0; i < 8; i++) env.tick();
assert.ok(
  env.diagnostics().lastFrame.amp < loud,
  "speech energy must follow real audio",
);
assert.ok(env.diagnostics().lastFrame.audio < loudAudio);
env.window.AstaViews.show("chat");
env.frame();
assert.equal(env.node("avatarFrame").parentNode.id, "avatarCompactSlot");
assert.equal(env.diagnostics().lastFrame.compact, true);
assert.equal(env.diagnostics().frameEvery, 2);
env.window.AstaAvatar.setState({ mode: "thinking" });
env.frame();
env.frame();
assert.equal(env.diagnostics().lastFrame.mode, "thinking", "chat avatar keeps animating while thinking");
assert.ok(env.stats().raf, "chat avatar stays scheduled");
env.window.AstaAvatar.setState({ mode: "idle" });
env.frame();
assert.equal(env.diagnostics().frameEvery, 2);
env.window.AstaViews.show("home");
env.window.AstaAvatar.setState({
  mode: "speaking",
  model_services: { coding: { status: "generating" } },
});
env.frame();
assert.equal(
  env.diagnostics().frameEvery,
  2,
  "coding reduces the visual frame budget",
);
env.document.hidden = true;
env.docListeners.visibilitychange();
assert.equal(env.stats().raf + env.stats().timers, 0);
env.document.hidden = false;
env.docListeners.visibilitychange();
env.frame();
assert.ok(env.stats().raf);
env.node("avatarMotionToggle").onclick();
env.frame();
assert.equal(env.stats().raf + env.stats().timers, 0);
env.listeners.pagehide();
assert.equal(env.stats().disposals, 1);
assert.equal(env.stats().raf + env.stats().timers, 0);
const still = environment(false, true);
still.frame();
still.window.AstaAvatar.setReady(true);
still.frame();
assert.equal(still.stats().raf + still.stats().timers, 0);
assert.equal(still.node("avatarMotionToggle").disabled, true);
const fallback = environment(true);
fallback.frame();
assert.equal(fallback.diagnostics().renderer, "canvas2d");
assert.equal(fallback.node("avatarFallback").hidden, false);
console.log(
  "Reactive avatar unit checks passed: budgets, audio response, shared scene, visibility, reduced motion, cleanup and fallback.",
);
