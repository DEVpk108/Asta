from types import SimpleNamespace
from core import Kernel
from core.contracts import ToolRequest
from core.tools.approval import ApprovalManager
from hud.hud_module import HUDModule
from tests.test_hud_module import RecordingHUDTransport


def test_approval_buttons_emit_only_live_request_ids_and_boolean_decisions():
    kernel = Kernel()
    hud = HUDModule(kernel); hud.transport = RecordingHUDTransport()
    request = ToolRequest(tool='system.run_command', arguments={'target':'python','arguments':['main.py'],'cwd':'.'}, request_id='review-me')
    kernel.approval_manager.request_approval(request, 'confirm')
    sent = []
    kernel.event_bus.subscribe('tool_confirmation_response', lambda **event: sent.append(event))
    hud._publish_state()
    assert hud.state.approvals[0]['can_approve'] is True
    assert 'main.py' in hud.state.approvals[0]['preview']
    hud.on_transport_message({'type':'hud.approval','request_id':'missing','approved':True})
    hud.on_transport_message({'type':'hud.approval','request_id':'review-me','approved':'true'})
    assert not sent
    hud.on_transport_message({'type':'hud.approval','request_id':'review-me','approved':True})
    assert sent == [{'request_id':'review-me','approved':True}]
    kernel.shutdown()


def test_expired_approval_never_emits_a_confirmation(monkeypatch):
    kernel = Kernel(); kernel.approval_manager = ApprovalManager(ttl_seconds=1)
    hud = HUDModule(kernel); hud.transport = RecordingHUDTransport()
    request = ToolRequest(tool='system.run_command', arguments={'target':'python'}, request_id='expired')
    pending = kernel.approval_manager.request_approval(request, 'confirm')
    monkeypatch.setattr('core.tools.approval.time.monotonic', lambda: pending.created_at + 2)
    sent=[]; kernel.event_bus.subscribe('tool_confirmation_response', lambda **event: sent.append(event))
    hud.on_transport_message({'type':'hud.approval','request_id':'expired','approved':True})
    assert not sent
    assert hud.state.approvals == []
    kernel.shutdown()


def test_unpreviewed_tool_cannot_be_approved_from_redesigned_hud():
    kernel = Kernel(); hud = HUDModule(kernel); hud.transport = RecordingHUDTransport()
    request = ToolRequest(tool='computer.type_text', arguments={'text':'PRIVATE_PAYLOAD'}, request_id='private')
    kernel.approval_manager.request_approval(request, 'confirm')
    sent=[]; kernel.event_bus.subscribe('tool_confirmation_response', lambda **event: sent.append(event))
    hud._publish_state()
    assert 'PRIVATE_PAYLOAD' not in str(hud.state.approvals)
    assert hud.state.approvals[0]['can_approve'] is False
    hud.on_transport_message({'type':'hud.approval','request_id':'private','approved':True})
    assert not sent
    hud.on_transport_message({'type':'hud.approval','request_id':'private','approved':False})
    assert sent == [{'request_id':'private','approved':False}]
    kernel.shutdown()


def test_project_steps_never_publish_code_or_model_rationale():
    from tests.test_workspace_project_task import _happy_plan, FixedPlanBrain, GOAL
    from core import Planner
    from core.intent_router import IntentRouter
    from core.tools import ListWorkspaceFilesTool, ReadWorkspaceFileTool, WriteWorkspaceFileTool, RunCommandTool
    kernel=Kernel()
    for tool in (ListWorkspaceFilesTool(kernel.workspace_manager), ReadWorkspaceFileTool(kernel.workspace_manager), WriteWorkspaceFileTool(kernel.workspace_manager), RunCommandTool(kernel.workspace_manager)):
        kernel.register_tool(tool)
    planner=Planner(kernel.tool_registry, agent_brain=FixedPlanBrain(_happy_plan()))
    plan=planner.plan(GOAL, intent=IntentRouter().analyze(GOAL))
    task=kernel.task_manager.create(GOAL, plan=plan)
    task.metadata['rationale']='PRIVATE_REASONING'
    kernel.background_tasks=SimpleNamespace(snapshot=lambda:{'task_id':task.id})
    hud=HUDModule(kernel);hud.transport=RecordingHUDTransport();hud._publish_state()
    assert any(item['tool'] == 'filesystem.write_file' for item in hud.state.project_steps)
    assert all(set(item) == {'id','tool','status','path','verification'} for item in hud.state.project_steps)
    assert 'ASTA_OK' not in str(hud.state.project_steps)
    assert 'PRIVATE_REASONING' not in str(hud.state.project_steps)
    kernel.background_tasks=None;kernel.shutdown()
