import re

from core.contracts import IntentResult, IntentType


_MIXED_SCREENSHOT_PATTERN = re.compile(
    r"^(?P<prompt>.+?)\s*(?:,?\s+)(?:and\s+then|then|after that|followed by|and)\s+"
    r"(?:please\s+)?(?:take(?:\s+(?:a|the))?\s+(?:a\s+)?(?:screenshot|screen\s*shot)"
    r"|capture(?:\s+(?:a|the))?\s+(?:a\s+)?(?:screenshot|screen\s*shot)"
    r"|screenshot|screen\s*shot)$",
    re.IGNORECASE,
)


def apply_final_runtime_patch():
    """Ensure conversational text is handled before the broad screenshot matcher."""
    from ai.ai_module import AIModule

    if getattr(AIModule, "_asta_final_runtime_patch_applied", False):
        return

    original = AIModule.on_user_message

    def patched(self, text):
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
                        self._generate_response(prompt)

                        request = self.tool_request_builder.build(
                            IntentResult(
                                intent=IntentType.COMMAND,
                                confidence=1.0,
                                normalized_text="take a screenshot",
                                entities={"action": "screenshot"},
                                requires_tools=True,
                                classifier="final_runtime_patch",
                            )
                        )
                        request.metadata["post_response_action"] = True
                        print(
                            f"[AI] Selected post-response tool: {request.tool} "
                            f"(request_id={request.request_id})",
                            flush=True,
                        )
                        self.event_bus.emit("tool_request", request=request)
                        return

        return original(self, text)

    AIModule.on_user_message = patched
    AIModule._asta_final_runtime_patch_applied = True
