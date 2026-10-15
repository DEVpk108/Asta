"""Legacy imports only; native AI/Voice handlers own all former patch behavior."""
from .post_response_capture import extract_prompt as _extract_post_response_screenshot
from .message_shortcuts import (_is_creator_identity_question,
    _is_screenshot_capture_request, _is_screenshot_open_request)
from .response_formatter import (_task_for_tool_result, _is_intermediate_task_result,
    _is_superseded_failure, _verified_visual_response)


def apply_ai_runtime_patch():
    """Compatibility no-op. No optional audio import and no class mutation."""
    return None
