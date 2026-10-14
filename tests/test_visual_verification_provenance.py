"""Only current, successful semantic inspection can satisfy a visual goal."""
import copy
import json
from types import SimpleNamespace

import pytest

from core import Kernel
from core.agent import AgentBrain
from core.contracts import Plan, PlanStep, TaskStatus, ToolDefinition, ToolRequest, ToolResult
from core.task_runtime import TaskRuntimeModule
from core.tools.vision_inspect import VisionInspectTool
from tests.test_agent_brain import FakeProvider, ToolDefinitionBackedTool


def _observation(source='vision.inspect', output=None, **extra):
    return {'kind': 'tool_result', 'source': source, 'data': {
        'success': True, 'output': output if output is not None else {'verified': True, 'visual_match': True},
        'request_id': 'inspection-1', 'plan_step_id': 'step-1', **extra,
    }}


def _state(observation=None):
    observation = observation or _observation()
    return {'goal': 'Verify calculator is visible', 'success_conditions': ['Calculator window is visible.'],
        'observations': [observation], 'latest_result': {
            'tool': observation['source'], 'success': observation['data']['success'],
            'output': copy.deepcopy(observation['data']['output']),
            'request_id': observation['data']['request_id'], 'plan_step_id': 'step-1',
        }}


@pytest.mark.parametrize('source', ['system.open_application', 'filesystem.read_file',
    'vision.screenshot', 'vision.locate', 'vision.open_screenshot', 'vision.untrusted'])
def test_positive_non_inspector_flags_do_not_verify_goal(source):
    assert not AgentBrain._has_verified_visual_observation(_state(_observation(source)))


@pytest.mark.parametrize('source', ['computer.click', 'computer.type_text', 'system.open_application',
    'vision.locate', 'vision.screenshot', 'filesystem.read_file'])
def test_old_success_is_invalidated_by_a_later_tool_observation(source):
    state = _state(_observation(source, {'verified': False}))
    state['observations'].insert(0, _observation())
    assert not AgentBrain._has_verified_visual_observation(state)


@pytest.mark.parametrize('value', [False, None, 'false', 'true', 1, [], {'flag': True}])
def test_verifier_approval_requires_boolean_true(value):
    state = _state(_observation(output={'verified': value, 'visual_match': True}))
    assert not AgentBrain._has_verified_visual_observation(state)


@pytest.mark.parametrize('output', [
    {'verified': False, 'analysis': {'verified': True, 'visual_match': True}},
    {'verified': False, 'visual_match': True, 'confidence': 0.2},
    {'verified': True, 'visual_match': False},
    {'analysis': {'verified': True}},
    {'window_detected': True},
    {'visible': True},
    {'visual_match': True},
])
def test_auxiliary_flags_cannot_override_canonical_verifier_outcome(output):
    assert not AgentBrain._has_verified_visual_observation(_state(_observation(output=output)))


@pytest.mark.parametrize('invalid', [None, 'bad observation', {}, {'source': 'vision.inspect'},
    {'source': 'vision.inspect', 'data': []}])
def test_malformed_latest_observation_does_not_revive_old_success(invalid):
    state = _state()
    state['observations'].append(invalid)
    assert not AgentBrain._has_verified_visual_observation(state)


@pytest.mark.parametrize('field,value', [('tool', 'computer.click'), ('success', False),
    ('request_id', 'different-request'), ('plan_step_id', 'different-step'),
    ('output', {'verified': False})])
def test_latest_result_must_agree_with_inspection_provenance(field, value):
    state = _state()
    state['latest_result'][field] = value
    assert not AgentBrain._has_verified_visual_observation(state)


def test_failed_inspection_cannot_verify_even_with_positive_output():
    state = _state(_observation(success=False))
    assert not AgentBrain._has_verified_visual_observation(state)


def test_missing_latest_result_does_not_allow_unbound_historical_evidence():
    state = _state()
    state.pop('latest_result')
    assert not AgentBrain._has_verified_visual_observation(state)


def test_current_inspection_is_accepted_without_mutating_state():
    state = _state()
    original = copy.deepcopy(state)
    assert AgentBrain._has_verified_visual_observation(state)
    assert state == original


def test_action_history_after_inspection_invalidates_the_observation():
    state = _state()
    state['actions'] = [{'tool': 'vision.inspect', 'outcome': 'success'},
                        {'tool': 'computer.click', 'outcome': 'success'}]
    assert not AgentBrain._has_verified_visual_observation(state)


def test_runtime_records_inspection_request_and_step_identity():
    task = SimpleNamespace(metadata={'agent_state': {}})
    TaskRuntimeModule._record_agent_observation(task, ToolResult(
        success=True, tool='vision.inspect', output={'verified': True},
        metadata={'request_id': 'req-real', 'plan_step_id': 'step-real'}))
    data = task.metadata['agent_state']['observations'][-1]['data']
    assert data['request_id'] == 'req-real'
    assert data['plan_step_id'] == 'step-real'


def _brain(kernel):
    provider = FakeProvider(json.dumps({'goal_satisfied': True, 'needs_observation': False,
        'needs_user': False, 'rationale': 'Model claims completion.', 'confidence': 1.0,
        'uncertainty': 0.0}))
    return AgentBrain(kernel, provider=provider, enabled=True)


def _register_inspector(kernel):
    kernel.register_tool(ToolDefinitionBackedTool(ToolDefinition(
        name='vision.inspect', description='Inert semantic inspector',
        input_schema={'type': 'object', 'properties': {'prompt': {'type': 'string'}},
                      'required': ['prompt']},
        metadata={'actions': ['inspect'], 'category': 'vision'})))


@pytest.mark.parametrize('confidence,expected', [(0.69, False), (0.70, True), (0.95, True)])
def test_real_inspector_confidence_gate_is_not_overridden_by_model_flags(tmp_path, confidence, expected):
    kernel = Kernel()
    engine = SimpleNamespace(inspect=lambda *args, **kwargs: {'json': {
        'visual_match': True, 'confidence': confidence, 'summary': 'Calculator visible'},
        'text': 'Calculator visible'})
    inspector = VisionInspectTool(engine, capture=lambda: tmp_path/'inert.png',
                                 media_session_reader=lambda *args: {})
    kernel.register_tool(inspector)
    result = inspector.execute(ToolRequest(tool='vision.inspect', request_id='inspection-1',
        arguments={'prompt': 'Verify calculator is visible'}))
    state = _state(_observation(output=result.output))
    try:
        assert _brain(kernel).decide(state).goal_satisfied is expected
    finally:
        kernel.shutdown()


def test_real_runtime_requests_fresh_inspection_instead_of_completing_after_click(monkeypatch):
    monkeypatch.setenv('ASTA_AGENT_POST_ACTION', 'always')
    kernel = Kernel()
    _register_inspector(kernel)
    kernel.agent_brain = _brain(kernel)
    tasks = TaskRuntimeModule(kernel)
    tasks.initialize()
    requests, verified = [], []
    kernel.event_bus.subscribe('tool_request', lambda request: requests.append(request))
    kernel.event_bus.subscribe('task_goal_verified', lambda **payload: verified.append(payload))
    step = PlanStep(id='click', description='click another window', metadata={
        'tool': 'computer.click', 'action': 'click'})
    task = kernel.task_manager.create('Verify calculator is visible', plan=Plan(
        goal='Verify calculator is visible', steps=[step]), pending_steps=[step.description],
        metadata={'agent_state': {'success_conditions': ['Calculator window is visible.'],
            'observations': [_observation()], 'actions': []}})
    try:
        kernel.event_bus.emit('tool_result', result=ToolResult(
            success=True, tool='computer.click', output={'verified': True},
            metadata={'task_id': task.id, 'plan_step_id': step.id,
                      'task_step': step.description, 'request_id': 'click-1'}))
        assert task.status is TaskStatus.ACTIVE
        assert verified == []
        assert len(requests) == 1 and requests[0].tool == 'vision.inspect'
        assert task.evidence[-1]['agent_decision']['goal_satisfied'] is False
    finally:
        tasks.shutdown()
        kernel.shutdown()


@pytest.mark.parametrize('value', ['false', 'true', 1])
def test_deterministic_runtime_does_not_complete_from_truthy_non_boolean_verification(value):
    kernel = Kernel()
    tasks = TaskRuntimeModule(kernel)
    task = kernel.task_manager.create('Verify calculator is visible',
        metadata={'agent_state': {'success_conditions': ['Calculator window is visible.']}})
    try:
        assert not tasks._complete_from_verified_visual_result(task,
            ToolResult(success=True, tool='vision.inspect', output={'verified': value}),
            evidence={}, plan_step_id=None, step_description='inspect')
        assert task.status is TaskStatus.ACTIVE
    finally:
        kernel.shutdown()
