from __future__ import annotations
import json
import pytest
from core import Kernel, Planner
from core.intent_router import IntentRouter
from core.agent.brain import AgentBrain
from core.planner import PlanningError
from tests.test_workspace_project_task import FixedPlanBrain, _happy_plan, _build_runtime, _shutdown, GOAL


def test_project_intent_keeps_exact_literals_and_paths():
    goal = 'Create MixedCase.py containing only print("ASTA_OK"). Run it.'
    intent = IntentRouter().analyze(goal)
    assert intent.entities['goal'] == goal
    assert intent.normalized_text == goal


def test_prompt_declares_verified_readback_and_equal_decoded_content():
    kernel = Kernel()
    brain = AgentBrain(kernel, enabled=True)
    intent = IntentRouter().analyze(GOAL)
    payload = json.loads(brain._build_prompt(GOAL, intent))
    example = payload['project_contract_example']
    # The content is written once; ASTA copies it into the read-back check.
    assert 'expected_content' not in example[1]
    assert "Write each file's content ONCE" in payload['instruction']
    assert example[1]['verification'] == 'filesystem.file_content'
    assert example[1]['tool'] == 'filesystem.read_file'
    assert 'ASTA_OK' in payload['intent']['goal']
    assert 'BEFORE executing code' in payload['instruction']
    kernel.shutdown()


def test_missing_readback_is_added_without_asking_the_model_again(tmp_path):
    good = _happy_plan()
    bad = [step for step in good if step.get('tool') != 'filesystem.read_file']
    workspace = tmp_path/'project'; workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path/'authority.json', bad, authorize_run=False)
    brain = FixedPlanBrain(bad, replanned_steps=good)
    kernel.planner.agent_brain = brain
    try:
        plan = kernel.planner.plan(GOAL, intent=IntentRouter().analyze(GOAL))
        assert brain.plan_calls == 1
        write = next(step for step in plan.steps if step.metadata.get('tool') == 'filesystem.write_file')
        readback = plan.steps[plan.steps.index(write) + 1]
        assert readback.metadata['tool'] == 'filesystem.read_file'
        assert readback.metadata['expected_content'] == write.metadata['content']
        assert not (workspace/'main.py').exists()
    finally:
        _shutdown(tasks, ai, tools); kernel.shutdown()


def test_repeated_invalid_verification_stops_after_two_proposals(tmp_path):
    bad = []
    for step in _happy_plan():
        if step.get('tool') == 'filesystem.read_file':
            continue
        if step.get('tool') == 'filesystem.write_file':
            # A content check on the write itself is not a read-back.
            step = {**step, 'verification': 'filesystem.file_content'}
        bad.append(step)
    workspace = tmp_path/'project'; workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path/'authority.json', bad)
    brain = FixedPlanBrain(bad)
    kernel.planner.agent_brain = brain
    try:
        with pytest.raises(PlanningError, match='File-content verification'):
            kernel.planner.plan(GOAL, intent=IntentRouter().analyze(GOAL))
        assert brain.plan_calls == 2
        assert not (workspace/'main.py').exists()
    finally:
        _shutdown(tasks, ai, tools); kernel.shutdown()


def test_incomplete_agent_output_never_returns_a_partial_plan(monkeypatch):
    from tests.test_concurrent_models import dual
    from tests.test_model_router import FakeEngine
    from ai.model_router import RoutedLLMProvider, ModelRoutingError
    kernel, runtime, managers, log = dual(monkeypatch)
    class IncompleteEngine(FakeEngine):
        def generate_response(self, *args, **kwargs):
            self.last_generation = {'cancelled': True, 'timed_out': True}
            return '{"steps": []}'
    client = RoutedLLMProvider(runtime, purpose='agent', engine_factory=IncompleteEngine)
    try:
        with pytest.raises(ModelRoutingError, match='timeout'):
            client.generate_for_task('plan', goal=GOAL, coding_scope=True)
    finally:
        runtime.shutdown()
