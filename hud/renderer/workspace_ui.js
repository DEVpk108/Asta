"use strict";
(function () {
  var bridge = window.asta || null;
  var $ = function (id) {
    return document.getElementById(id);
  };
  var ready = false;
  var latest = {};
  var pending = new Map();
  var phases = {
    queued: ["Queued", "Waiting for the project worker."],
    planning: [
      "Planning",
      "Preparing a proposal. No actions are authorized by this stage.",
    ],
    planned: ["Plan ready", "The runtime has validated the proposed steps."],
    working: ["Working", "Following the validated project plan."],
    executing: ["Executing", "Running the current tool step."],
    awaiting_approval: [
      "Needs approval",
      "Review the exact requested action below.",
    ],
    replanning: [
      "Replanning",
      "Preparing a corrected plan within runtime limits.",
    ],
    recovery: ["Recovering", "Evaluating a bounded recovery step."],
    completed: ["Completed", "The runtime reports the project task completed."],
    failed: [
      "Failed",
      "The project task failed. Ask for status before retrying.",
    ],
    cancelled: [
      "Cancelled",
      "Future steps stopped. An in-flight tool may still finish; edits are not undone.",
    ],
    paused: ["Paused", "The project task is paused."],
    testing: ["Testing", "Checking the project through its registered tools."],
  };
  function text(id, value) {
    var node = $(id);
    if (node) node.textContent = String(value || "");
  }
  function setReady(value, label) {
    ready = !!value;
    if (window.AstaAvatar) window.AstaAvatar.setReady(ready);
    text("engineLabel", label || (ready ? "Connected to ASTA" : "Connecting"));
    $("engineLed").style.background = ready ? "var(--green)" : "var(--muted)";
    [
      "chatInput",
      "chatSend",
      "chatConversation",
      "projectStatus",
      "chatNew",
    ].forEach(function (id) {
      $(id).disabled = !ready;
    });
    if (!ready) {
      text("connectionDetail", "Runtime unavailable — actions disabled");
      text("statusMode", "Unavailable");
      text("voiceDetail", "Waiting for the runtime");
      $("projectCancel").disabled = true;
      pending.forEach(function (value) {
        value.card.querySelectorAll("button").forEach(function (button) {
          button.disabled = true;
        });
      });
    }
  }
  function service(name, state) {
    var value = (state.model_services || {})[name];
    if (!value && state.model_profile === name)
      value = { status: state.model_status, model: state.model_name };
    var status = value ? String(value.status || "unknown") : "unknown";
    text(name + "Status", status === "stopped" ? "Unloaded" : status);
    $(name + "Status").dataset.status = status;
    text(name + "Model", value && value.model ? value.model : "Not reported");
  }
  function renderSteps(steps) {
    var list = $("projectSteps");
    list.replaceChildren();
    (Array.isArray(steps) ? steps : []).forEach(function (step, index) {
      var item = document.createElement("li");
      item.dataset.status = String(step.status || "pending");
      var marker = document.createElement("span");
      marker.className = "step-marker";
      marker.textContent =
        step.status === "completed"
          ? "✓"
          : step.status === "failed"
            ? "!"
            : String(index + 1);
      marker.setAttribute("aria-hidden", "true");
      var detail = document.createElement("div");
      var label = document.createElement("span");
      label.className = "step-label";
      var names = {
        "filesystem.write_file": "Write file",
        "filesystem.read_file": "Read and verify file",
        "filesystem.list_files": "Inspect workspace",
        "system.run_command": "Run in sandbox",
      };
      label.textContent =
        step.tool === "filesystem.read_file" && !step.verification
          ? "Read file"
          : names[step.tool] || step.tool || "Tool step";
      var secondary = document.createElement("span");
      secondary.className = "step-detail";
      secondary.textContent =
        (step.path ? step.path + " · " : "") + String(step.status || "pending");
      detail.append(label, secondary);
      item.append(marker, detail);
      list.appendChild(item);
    });
  }
  function renderApprovals(approvals) {
    var region = $("approvalRegion");
    var ids = new Set();
    (Array.isArray(approvals) ? approvals : []).forEach(function (request) {
      var id = String(request.request_id || "");
      if (!id) return;
      ids.add(id);
      if (pending.has(id)) return;
      var card = document.createElement("article");
      card.className = "approval-card";
      var kicker = document.createElement("span");
      kicker.className = "eyebrow";
      kicker.textContent = "YOUR PERMISSION";
      var title = document.createElement("h3");
      title.textContent = "Review this action";
      var desc = document.createElement("p");
      desc.textContent = String(request.tool || "Tool request");
      var preview = document.createElement("pre");
      preview.textContent = String(request.preview || "");
      var note = document.createElement("p");
      note.textContent =
        request.tool === "system.run_command"
          ? "Sandbox execution · read-only project snapshot · network disabled."
          : "A command preview is not available for this tool. Use the existing confirmation flow; you can reject here.";
      var clock = document.createElement("span");
      clock.className = "approval-clock";
      var actions = document.createElement("div");
      actions.className = "approval-actions";
      [
        { label: "Approve action", approved: true },
        { label: "Reject", approved: false },
      ].forEach(function (choice) {
        var button = document.createElement("button");
        button.type = "button";
        button.className = choice.approved ? "approve-button" : "quiet-button";
        button.textContent = choice.label;
        button.disabled =
          !ready || (choice.approved && request.can_approve !== true);
        button.addEventListener("click", function () {
          if (
            !ready ||
            (choice.approved && request.can_approve !== true) ||
            !bridge ||
            !bridge.respondToApproval ||
            (request.expires_at && Date.now() >= request.expires_at * 1000)
          )
            return;
          actions.querySelectorAll("button").forEach(function (node) {
            node.disabled = true;
          });
          clock.textContent =
            "Response sent. Waiting for runtime confirmation…";
          bridge.respondToApproval(id, choice.approved);
          window.requestAnimationFrame(function () {
            card.scrollIntoView({ block: "nearest", behavior: "auto" });
          });
        });
        actions.appendChild(button);
      });
      card.append(kicker, title, desc, preview, note, clock, actions);
      region.appendChild(card);
      // Bring the real approval into view, not a decorative progress animation.
      window.requestAnimationFrame(function () {
        card.scrollIntoView({ block: "nearest", behavior: "auto" });
      });
      pending.set(id, { request: request, card: card, clock: clock });
    });
    pending.forEach(function (value, id) {
      if (!ids.has(id)) {
        value.card.remove();
        pending.delete(id);
      }
    });
    updateApprovalClocks();
  }
  function updateApprovalClocks() {
    pending.forEach(function (value) {
      var expires = value.request.expires_at;
      var seconds = expires
        ? Math.max(0, Math.ceil(expires - Date.now() / 1000))
        : null;
      if (seconds === 0) {
        value.clock.textContent =
          "Approval expired. Check project status before retrying.";
        value.card.querySelectorAll("button").forEach(function (button) {
          button.disabled = true;
        });
      } else if (!value.card.querySelector("button").disabled)
        value.clock.textContent =
          seconds === null
            ? "Request-bound approval"
            : "Expires in " + seconds + " seconds";
    });
  }
  function stateChanged(state) {
    latest = state || {};
    if (window.AstaAvatar) window.AstaAvatar.setState(latest);
    var mode = String(latest.mode || "idle");
    text(
      "statusMode",
      mode === "idle" ? "Ready" : mode.charAt(0).toUpperCase() + mode.slice(1),
    );
    var voiceDetail = {
      idle: "Ask by voice or text",
      listening: "Listening for your message",
      thinking: "Preparing a response",
      speaking: "Speaking",
      executing: "Tool step in progress",
      approval: "Action needs your approval",
      error: "Check runtime status",
    };
    text("voiceDetail", voiceDetail[mode] || "");
    $("voiceIndicator").style.background =
      mode === "error"
        ? "var(--red)"
        : mode === "approval"
          ? "var(--amber)"
          : "var(--blue)";
    service("general", latest);
    service("coding", latest);
    var background = latest.background_task;
    var phase = background ? String(background.phase || "") : "";
    var display = phases[phase] || [
      phase.replace(/_/g, " ") || "No task",
      background
        ? "See runtime status for this task."
        : "Start a project task to see its validated steps here.",
    ];
    text("backgroundTaskStatus", display[0]);
    $("backgroundTaskStatus").dataset.status = phase;
    text("projectDetail", display[1]);
    renderSteps(latest.project_steps);
    $("projectCancel").disabled =
      !ready ||
      !background ||
      ["completed", "failed", "cancelled"].indexOf(phase) !== -1;
    renderApprovals(latest.approvals);
    var task = $("taskStatus");
    task.hidden =
      !!background ||
      ["executing", "thinking", "approval", "error"].indexOf(mode) === -1;
    text("taskStatusStage", latest.status || mode);
    text(
      "taskStatusDetail",
      mode === "approval"
        ? "Review the pending request."
        : latest.activity || "Runtime activity",
    );
    text(
      "connectionDetail",
      latest.model_mode === "on_demand"
        ? "On-demand models · chat may wait during coding"
        : latest.model_mode === "dual"
          ? "Dual model services · one project task at a time"
          : "Local model services",
    );
    if (!ready) setReady(false);
  }
  function workspaceChanged(workspace) {
    workspace = workspace || {};
    var valid = !!workspace.project_name;
    $("workspaceContext").hidden = !valid;
    $("workspaceEmpty").hidden = valid;
    text("workspaceProjectName", workspace.project_name);
    text("workspaceBranch", workspace.branch);
    $("workspaceBranch").hidden = !workspace.branch;
    text("workspaceRepository", workspace.repository);
    $("workspaceRepository").hidden = !workspace.repository;
    var files = $("workspaceRecentFiles");
    files.replaceChildren();
    (Array.isArray(workspace.recent_files)
      ? workspace.recent_files
      : []
    ).forEach(function (file) {
      var node = document.createElement("span");
      node.className = "workspace-file";
      node.textContent = file;
      files.appendChild(node);
    });
    $("workspaceRecentRow").hidden = !files.children.length;
  }
  if (bridge && bridge.platform === "darwin")
    document.body.classList.add("is-mac");
  if (!bridge) document.querySelector(".tb-right").hidden = true;
  [
    ["wcMin", "minimize"],
    ["wcMax", "toggleMaximize"],
    ["wcClose", "close"],
  ].forEach(function (pair) {
    $(pair[0]).addEventListener("click", function () {
      if (bridge && bridge[pair[1]]) bridge[pair[1]]();
    });
  });
  $("projectStatus").addEventListener("click", function () {
    if (ready && bridge) bridge.sendTextMessage("project task status");
  });
  $("projectCancel").addEventListener("click", function () {
    if (ready && bridge) bridge.sendTextMessage("cancel project task");
  });
  document.querySelectorAll("[data-prompt]").forEach(function (button) {
    button.addEventListener("click", function () {
      $("chatInput").value = button.dataset.prompt;
      $("chatInput").focus();
    });
  });
  $("themeToggle").addEventListener("click", function () {
    var dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    $("themeToggle").setAttribute(
      "aria-label",
      dark ? "Switch to dark theme" : "Switch to light theme",
    );
  });
  window.AstaHUD = {
    applyCanonicalState: stateChanged,
    beginRuntime: function () {},
    setAudioLevel: function (level) {
      var value = Math.max(0, Math.min(1, Number(level) || 0));
      if (window.AstaAvatar) window.AstaAvatar.setAudioLevel(value);
      $("waveform")
        .querySelectorAll("i")
        .forEach(function (bar, index) {
          bar.style.height =
            4 + value * (index === 2 ? 18 : index % 2 ? 12 : 6) + "px";
        });
    },
  };
  setReady(false);
  if (bridge) {
    if (bridge.onHudLifecycle)
      bridge.onHudLifecycle(function (event) {
        setReady(
          event && event.status === "ready",
          event && event.status === "stopping" ? "Stopping ASTA" : undefined,
        );
        if (ready) stateChanged(latest);
      });
    if (bridge.onHudState) bridge.onHudState(stateChanged);
    if (bridge.onHudWorkspace) bridge.onHudWorkspace(workspaceChanged);
    if (bridge.onHudAudio)
      bridge.onHudAudio(function (event) {
        window.AstaHUD.setAudioLevel(event && event.level);
      });
  }
  window.setInterval(updateApprovalClocks, 1000);
})();
