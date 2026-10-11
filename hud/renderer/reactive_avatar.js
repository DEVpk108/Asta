"use strict";
(function () {
  var $ = function (id) {
    return document.getElementById(id);
  };
  var panel = $("chatPanel"),
    frame = $("avatarFrame");
  if (!panel || !frame) return;
  var view = "home";
  var bridge = window.asta || null;
  var preferences = window.matchMedia("(prefers-reduced-motion: reduce)");
  var motion = true,
    ready = false,
    disposed = false;
  var mode = "idle",
    codingActive = false,
    audioTarget = 0,
    audioLevel = 0,
    audioAt = 0;
  var engine = null,
    fallback = false,
    raf = 0,
    timer = 0,
    lastAt = 0,
    visualTime = 0,
    liveAmp = 0.14;
  var drawCount = 0,
    lastFrame = null;
  var resizeObserver = null;
  var descriptions = {
    idle: ["Ready", "Ask by voice or text."],
    listening: ["Listening", "Your voice channel is active."],
    thinking: [
      "Preparing a response",
      "The runtime reports foreground processing.",
    ],
    speaking: ["Speaking", "Movement follows speech audio output."],
    executing: [
      "Executing a step",
      "Follow the observed tool steps in project activity.",
    ],
    approval: [
      "Your approval is needed",
      "Review the pending action in project activity.",
    ],
    error: ["Needs attention", "Check the runtime status before continuing."],
  };
  function label() {
    var value = ready
      ? descriptions[mode] || descriptions.idle
      : ["Connecting", "Waiting for ASTA’s runtime."];
    $("homeStateLabel").textContent = value[0];
    $("homeStateDetail").textContent = value[1];
    frame.dataset.mode = ready ? mode : "offline";
    var button = $("avatarMotionToggle");
    button.disabled = preferences.matches;
    button.setAttribute(
      "aria-pressed",
      motion && !preferences.matches ? "true" : "false",
    );
    button.textContent = preferences.matches
      ? "Reduced motion enabled"
      : motion
        ? "Pause visual motion"
        : "Resume visual motion";
    $("avatarMotionLabel").textContent = preferences.matches
      ? "Reduced motion"
      : motion
        ? "Motion on"
        : "Motion paused";
  }
  function startFallback() {
    if (engine && engine.dispose) engine.dispose();
    engine = null;
    fallback = true;
    $("avatarCanvas").hidden = true;
    $("avatarFallback").hidden = false;
    try {
      engine = window.AstaFallback2D.create($("avatarFallback"), {
        maxDpr: 1.25,
        maxDimension: 768,
      });
    } catch (error) {
      engine = null;
    }
    $("avatarRenderMode").textContent = engine
      ? "2D fallback"
      : "Visual unavailable";
    frame.dataset.renderer = engine ? "canvas2d" : "unavailable";
  }
  function boot() {
    try {
      engine = window.AstaEngine.create($("avatarCanvas"), {
        shape: "humanoid",
        maxParticles: 12000,
        maxDpr: 1.25,
        maxDimension: 768,
        lowPower: true,
      });
      if (!engine || !engine.ok) throw new Error("Avatar renderer unavailable");
      $("avatarRenderMode").textContent = "3D particle avatar";
      frame.dataset.renderer = "webgl2";
    } catch (error) {
      startFallback();
    }
  }
  function stopScheduled() {
    if (raf) cancelAnimationFrame(raf);
    if (timer) clearTimeout(timer);
    raf = 0;
    timer = 0;
  }
  function continuous() {
    return (
      ready && motion && !preferences.matches && !document.hidden && !disposed
    );
  }
  function fps() {
    if (view === "chat") return mode === "speaking" ? 15 : 6;
    if (codingActive) return 12;
    return ["listening", "thinking", "speaking", "executing"].indexOf(mode) !==
      -1
      ? 30
      : 12;
  }
  function draw(now) {
    raf = 0;
    if (disposed || document.hidden || !engine) return;
    var moving = continuous();
    var dt = lastAt ? Math.min(0.15, Math.max(0, (now - lastAt) / 1000)) : 0.04;
    lastAt = now;
    if (moving) visualTime += dt;
    if (now - audioAt > 650) audioTarget = 0;
    var signal = ready && mode === "speaking" && moving ? audioTarget : 0;
    audioLevel += (signal - audioLevel) * Math.min(1, dt * 14);
    var base =
      {
        idle: 0.14,
        listening: 0.32,
        thinking: 0.4,
        speaking: 0.18,
        executing: 0.3,
        approval: 0.12,
        error: 0.1,
      }[mode] || 0.14;
    var target = ready ? base + audioLevel * 0.6 : 0.08;
    liveAmp += (target - liveAmp) * Math.min(1, dt * 8);
    var values = {
      time: moving ? visualTime : 0,
      progress: 1,
      amp: moving ? liveAmp : target,
      colorMix: mode === "approval" ? 0.85 : 0.42,
      rot: moving
        ? Math.sin(visualTime * 0.24) * (0.08 + audioLevel * 0.05)
        : 0,
      bloom: 0.75 + audioLevel * 0.3,
      compact: view === "chat",
    };
    try {
      engine.render(values);
    } catch (error) {
      if (!fallback) {
        startFallback();
        return refresh();
      }
      return;
    }
    drawCount++;
    lastFrame = values;
    if (continuous())
      timer = setTimeout(function () {
        timer = 0;
        raf = requestAnimationFrame(draw);
      }, 1000 / fps());
  }
  function refresh() {
    stopScheduled();
    lastAt = 0;
    if (!document.hidden && !disposed) raf = requestAnimationFrame(draw);
  }
  function setView(next, focus) {
    if (next !== "home" && next !== "chat") return;
    if (next !== view) {
      view = next;
      panel.classList.toggle("view-home", view === "home");
      panel.classList.toggle("view-chat", view === "chat");
      $("voiceHome").hidden = view !== "home";
      $("viewHome").hidden = view === "home";
      $("viewChat").hidden = view === "chat";
      $("viewTitle").textContent =
        view === "home" ? "Voice workspace" : "Conversation";
      $("chatFullscreen").hidden = view === "home";
      (view === "home"
        ? $("avatarHomeSlot")
        : $("avatarCompactSlot")
      ).appendChild(frame);
      frame.classList.toggle("compact", view === "chat");
      window.dispatchEvent(
        new CustomEvent("asta:view-change", { detail: { view: view } }),
      );
    }
    refresh();
    if (focus !== false)
      requestAnimationFrame(function () {
        (view === "chat" ? $("chatInput") : $("viewChat")).focus();
      });
  }
  window.AstaViews = {
    show: setView,
    current: function () {
      return view;
    },
  };
  window.AstaAvatar = {
    setReady: function (value) {
      ready = !!value;
      if (!ready) {
        audioTarget = 0;
        audioLevel = 0;
      }
      label();
      refresh();
    },
    setState: function (state) {
      var requested = String((state || {}).mode || "idle");
      mode = descriptions[requested] ? requested : "idle";
      var services = (state || {}).model_services || {};
      codingActive = !!(
        services.coding &&
        ["loading", "generating"].indexOf(services.coding.status) !== -1
      );
      label();
      refresh();
    },
    setAudioLevel: function (value) {
      var number = Number(value);
      audioTarget = Number.isFinite(number)
        ? Math.max(0, Math.min(1, number))
        : 0;
      audioAt = performance.now();
      // Do not restart scheduling for every telemetry sample, or replay stale audio.
    },
    diagnostics: function () {
      return {
        view: view,
        renderer: frame.dataset.renderer,
        mode: mode,
        ready: ready,
        motion: continuous(),
        fpsCap: fps(),
        drawCount: drawCount,
        particles: engine ? engine.count : 0,
        scheduled: !!(raf || timer),
        lastFrame: lastFrame,
      };
    },
  };
  $("viewChat").addEventListener("click", function () {
    setView("chat");
  });
  $("viewHome").addEventListener("click", function () {
    setView("home");
  });
  $("homeReadChat").addEventListener("click", function () {
    setView("chat");
  });
  $("avatarMotionToggle").addEventListener("click", function () {
    motion = !motion;
    label();
    refresh();
  });
  document.addEventListener("visibilitychange", function () {
    refresh();
  });
  preferences.addEventListener("change", function () {
    label();
    refresh();
  });
  $("avatarCanvas").addEventListener("webglcontextlost", function (event) {
    event.preventDefault();
    stopScheduled();
    startFallback();
    refresh();
  });
  window.addEventListener("resize", refresh);
  if (window.ResizeObserver) {
    resizeObserver = new ResizeObserver(refresh);
    resizeObserver.observe(frame);
  }
  window.addEventListener(
    "pagehide",
    function () {
      disposed = true;
      stopScheduled();
      if (resizeObserver) resizeObserver.disconnect();
      if (engine && engine.dispose) engine.dispose();
    },
    { once: true },
  );
  function caption(message) {
    if (!message || message.role === "user" || !message.text) return;
    var value = String(message.text)
      .replace(/```[\s\S]*?(?:```|$)/g, "[Code is available in chat]")
      .replace(/[*#`]/g, "")
      .trim();
    $("homeCaption").textContent =
      value.length > 240 ? value.slice(0, 237) + "…" : value;
  }
  if (bridge && bridge.onHudChat) bridge.onHudChat(caption);
  if (bridge && bridge.onHudChatHistory)
    bridge.onHudChatHistory(function (history) {
      var items = Array.isArray(history)
        ? history
        : (history && history.messages) || [];
      $("homeCaption").textContent =
        "Speak using your wake word, or send a message below.";
      for (var i = items.length - 1; i >= 0; i--)
        if (items[i].role !== "user") {
          caption(items[i]);
          break;
        }
    });
  // One scene is moved between views: no second WebGL context or duplicate subscriptions.
  boot();
  label();
  $("chatFullscreen").hidden = true;
  refresh();
})();
