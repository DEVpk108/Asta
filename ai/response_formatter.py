"""Tool-result wording and suppression policy; no runtime class replacement."""
import re
from core.contracts import ToolResult
from core.visual_evidence import is_verified_inspection_output

def _task_for_tool_result(ai, result):
    task_manager = getattr(ai.kernel, "task_manager", None)
    if task_manager is None or not isinstance(result, ToolResult):
        return None

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    task_id = metadata.get("task_id")
    try:
        task = (
            task_manager.get(task_id)
            if task_id
            else task_manager.current()
        )
    except Exception:
        return None
    return task


def _is_intermediate_task_result(ai, result) -> bool:
    task = _task_for_tool_result(ai, result)
    if task is None:
        return False

    plan = getattr(task, "plan", None)
    plan_step_id = (
        result.metadata.get("plan_step_id")
        if isinstance(result.metadata, dict)
        else None
    )
    steps = getattr(plan, "steps", None) or []
    if plan is None or not plan_step_id or not steps:
        return False

    for index, step in enumerate(steps):
        if getattr(step, "id", None) != plan_step_id:
            continue
        # A result from any step with a later step is an internal progress
        # update. Do not let synchronous nested EventBus dispatch turn it into
        # user-facing speech before the final step has produced its answer.
        return index < (len(steps) - 1)

    return False


def _is_superseded_failure(ai, result) -> bool:
    """A failed step that the task runtime already retried.

    Recovery retries run synchronously inside the same tool_result dispatch,
    so by the time this handler sees the original failure the retry may have
    succeeded (and been announced). Only speak a failure whose step is still
    marked failed.
    """
    if not isinstance(result, ToolResult) or result.success:
        return False
    task = _task_for_tool_result(ai, result)
    plan = getattr(task, "plan", None)
    step_id = result.metadata.get("plan_step_id") if isinstance(result.metadata, dict) else None
    if plan is None or not step_id:
        return False
    try:
        step = plan.get_step(step_id)
    except Exception:
        return False
    status = str(getattr(getattr(step, "status", None), "value", getattr(step, "status", ""))).lower()
    return status in {"ready", "running", "completed"}


def _verified_visual_response(result: ToolResult) -> str | None:
    if not result.success or result.tool != "vision.inspect":
        return None

    output = result.output if isinstance(result.output, dict) else {}
    if not is_verified_inspection_output(output):
        return None

    summary = " ".join(str(output.get("summary") or "").strip().split())
    if summary:
        return f"Done — {summary.rstrip(' .!?;:')}."

    return "Done — the requested visual condition is confirmed."


def format_tool_success(result: ToolResult) -> str:
    if result.tool == "vision.open_screenshot":
        return "Opened the latest screenshot."
    completion = (
        result.metadata.get("completion_message")
        if isinstance(result.metadata, dict)
        else None
    )
    if completion:
        return str(completion)
    output = result.output
    if isinstance(output, dict):
        target = output.get("target")
        if result.tool == "system.open_application" and target:
            return f"Opened {target}."
        if result.tool == "system.launch_application" and target:
            return f"Launched {target}."
        if result.tool == "system.close_application" and target:
            # Say what was actually closed ("Close sport" closed Spotify.exe).
            process = re.sub(r"\.exe$", "", str(output.get("process") or ""), flags=re.IGNORECASE)
            if process and process.lower().replace(" ", "") != str(target).lower().replace(" ", ""):
                return f"Closed {process}."
            return f"Closed {target}."
        if result.tool == "media.ui_play":
            return str(output.get("message") or "Playing it now.")
        if result.tool == "media.control":
            message = output.get("message")
            if message:
                return str(message)
            return "Media action completed."
        if result.tool == "system.stop_process" and output.get("pid"):
            return f"Stopped process {output['pid']}."
        if result.tool == "system.start_process" and target:
            return f"Started {target}."
        if result.tool == "vision.screenshot":
            path = output.get("path")
            if path:
                print(f"[AI] Screenshot saved: {path}", flush=True)
            return "Screenshot captured."
        if result.tool == "vision.inspect":
            summary = str(output.get("summary") or "").strip()
            if output.get("question") and summary:
                return summary
            if output.get("source") == "media_session" and output.get("track"):
                return f"Now playing {output['track']}."
            if is_verified_inspection_output(output):
                return "I checked the screen. The requested visual condition is confirmed."
            if bool(output.get("visual_match")):
                return "I checked the screen, but the visual verification was inconclusive."
            return "I checked the screen, but could not verify the requested visual condition."
        if result.tool == "notes.create_note":
            title = output.get("title")
            return f"Saved note '{title}'." if title else "Saved note."
        if result.tool == "notes.read_note":
            content = output.get("content")
            return str(content).strip() if content else "That note is empty."
        if result.tool == "notes.list_notes":
            count = int(output.get("count", 0))
            notes = output.get("notes") or []
            if count == 0:
                return "You don't have any saved notes yet."
            titles = [str(item.get("title")).strip() for item in notes[:8] if item.get("title")]
            suffix = f": {', '.join(titles)}" if titles else "."
            if count > len(titles):
                suffix = suffix.rstrip(".") + f", and {count - len(titles)} more."
            return f"You have {count} saved note{'s' if count != 1 else ''}{suffix}"
        if result.tool == "notes.search_notes":
            count = int(output.get("count", 0))
            matches = output.get("matches") or []
            if count == 0:
                return "I couldn't find any matching notes."
            titles = [str(item.get("title")).strip() for item in matches[:8] if item.get("title")]
            return f"I found {count} matching note{'s' if count != 1 else ''}: {', '.join(titles)}."
    if output is None:
        return f"{result.tool} completed successfully."
    return str(output)


def format_tool_failure(result: ToolResult, *, rejected_error, approval_expired_message) -> str:
    output = result.output if isinstance(result.output, dict) else {}

    if str(result.error or "") == rejected_error:
        return "Okay, I cancelled that."

    if str(result.error or "").startswith("Approval expired"):
        return approval_expired_message

    if result.tool == "system.close_application":
        target = output.get("target") or output.get("resolved_target")
        if target:
            return f"I couldn't close {target}."
        return "I couldn't close the application."

    if result.tool in {
        "system.open_application",
        "system.launch_application",
    }:
        target = output.get("target") or output.get("resolved_target")
        missing = re.search(
            r"No installed application matched '([^']+)'",
            str(result.error or ""),
        )
        if missing:
            return (
                f"I couldn't find an app called {missing.group(1)}. "
                "Could you say the name again?"
            )
        if target:
            return f"I couldn't open {target}."
        return "I couldn't open the application."

    if result.tool == "system.start_process":
        target = output.get("target")
        return f"I couldn't start {target}." if target else "I couldn't start the process."

    if result.tool == "media.ui_play":
        query = output.get("query") or "that"
        app = output.get("application") or "the app"
        if output.get("method"):
            return f"I pressed play on {query} in {app}, but it didn't start playing."
        return f"I couldn't find {query} in {app}'s results."

    if result.tool == "media.control":
        error = str(result.error or "").lower()
        if "one-time setup" in error or "not configured" in error or "asta_spotify_client_id" in error:
            return (
                "Spotify's developer API isn't set up yet. "
                "I'll open the setup in your browser if I can."
            )
        if "spotify authorization" in error or "authorize a.s.t.a" in error:
            return (
                "Spotify needs authorization. Please complete the Spotify authorization "
                "in the browser; I’ll continue the original task afterward."
            )
        return "I couldn't control media playback."

    if result.tool == "system.stop_process":
        return "I couldn't stop that process."

    if result.tool in {"vision.screenshot", "vision.open_screenshot"}:
        return "I couldn't complete the screenshot action."

    # Keep internal diagnostics in logs/HUD metadata, not in spoken audio.
    return "I couldn't complete that action."
