from uuid import uuid4

from core.contracts import IntentResult, IntentType, ToolRequest


class ToolRequestBuilder:
    """Convert tool-requiring intent results into ToolRequests.

    Intent analysis remains separate from execution. This builder only
    translates structured intent data into the executable tool contract.
    It does not inspect natural language or execute tools.
    """

    ACTION_TO_TOOL = {
        "open": "system.open_application",
        "close": "system.close_application",
        "launch": "system.launch_application",
        "start": "system.start_process",
        "run": "system.run_command",
        "stop": "system.stop_process",
        "screenshot": "vision.screenshot",
        "mute": "audio.mute",
        "unmute": "audio.unmute",
    }

    @classmethod
    def build(cls, intent: IntentResult) -> ToolRequest:
        if intent.intent != IntentType.COMMAND:
            raise ValueError(
                "ToolRequest can only be built from a command intent."
            )

        action = intent.entities.get("action")
        if not isinstance(action, str) or not action:
            raise ValueError(
                "Command intent is missing a valid action."
            )

        tool_name = cls.ACTION_TO_TOOL.get(action)
        if tool_name is None:
            raise ValueError(
                f"No tool mapping exists for command action: {action}"
            )

        arguments = dict(intent.entities)
        arguments.pop("action", None)

        return ToolRequest(
            tool=tool_name,
            arguments=arguments,
            request_id=str(uuid4()),
        )
