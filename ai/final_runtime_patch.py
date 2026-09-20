import re
import threading

from core.contracts import IntentResult, IntentType


_MIXED_SCREENSHOT_PATTERN = re.compile(
    r"^(?P<prompt>.+?)\s*(?:,?\s+)(?:and\s+then|then|after that|followed by|and)\s+"
    r"(?:please\s+)?(?:take(?:\s+(?:a|the))?\s+(?:a\s+)?(?:screenshot|screen\s*shot)"
    r"|capture(?:\s+(?:a|the))?\s+(?:a\s+)?(?:screenshot|screen\s*shot)"
    r"|screenshot|screen\s*shot)$",
    re.IGNORECASE,
)


_MIXED_SCREENSHOT_RENDER_DELAY = 0.20


def apply_final_runtime_patch():
    """Ensure mixed conversational requests capture the fully rendered HUD."""
    from ai.ai_module import AIModule

    if getattr(AIModule, "_asta_final_runtime_patch_applied", False):
        return

    original = AIModule.on_user_message

    def patched(self, text, runtime_context=None):
        if isinstance(text, str):
            normalized = " ".join(text.strip().split()).rstrip(" .!?;:")
            match = _MIXED_SCREENSHOT_PATTERN.match(normalized)
            if match:
                routed = self.kernel.intent_router.analyze(text)
                commands = routed.entities.get("commands") if routed.entities else None
                if not (
                    routed.intent == IntentType.COMMAND
                    and isinstance(commands, list)
                    and len(commands) >= 2
                ):
                    prompt = match.group("prompt").strip(" ,.!?;:")
                    if prompt:
                        print(
                            "[AI] Final mixed-request fix: conversational response + screenshot.",
                            flush=True,
                        )

                        screenshot_intent = IntentResult(
                            intent=IntentType.COMMAND,
                            confidence=1.0,
                            normalized_text="take a screenshot",
                            entities={"action": "screenshot"},
                            requires_tools=True,
                            classifier="final_runtime_patch",
                        )
                        request = self.tool_request_builder.build(screenshot_intent)
                        request.metadata["post_response_action"] = True

                        hud_seen = {"value": False}

                        def schedule_screenshot():
                            print(
                                "[AI] HUD rendered; waiting briefly before screenshot.",
                                flush=True,
                            )

                            def emit_request():
                                print(
                                    f"[AI] Selected post-response tool: {request.tool} "
                                    f"(request_id={request.request_id})",
                                    flush=True,
                                )
                                self.event_bus.emit("tool_request", request=request)

                            timer = threading.Timer(
                                _MIXED_SCREENSHOT_RENDER_DELAY,
                                emit_request,
                            )
                            timer.daemon = True
                            timer.start()

                        def on_hud_rendered(*_args, **_kwargs):
                            if hud_seen["value"]:
                                return
                            hud_seen["value"] = True
                            self.event_bus.unsubscribe("hud_rendered", on_hud_rendered)
                            schedule_screenshot()

                        self.event_bus.subscribe("hud_rendered", on_hud_rendered)
                        self._generate_response(prompt, runtime_context=None)

                        # Safety fallback for runtimes where the HUD isn't subscribed
                        # or doesn't emit hud_rendered.
                        if not hud_seen["value"]:
                            self.event_bus.unsubscribe("hud_rendered", on_hud_rendered)
                            hud_seen["value"] = True
                            schedule_screenshot()

                        return

        return original(self, text, runtime_context=runtime_context)

    AIModule.on_user_message = patched
    AIModule._asta_final_runtime_patch_applied = True
