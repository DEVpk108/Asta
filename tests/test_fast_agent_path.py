"""Fast command path: transcript cleanup, System 1 planning, lean agent loop."""

import threading
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from core import Kernel, Planner
from core.agent import AgentPlanProposal
from core.agent.brain import AgentBrain
from core.contracts import (
    ActionDecision,
    ActionType,
    IntentResult,
    IntentType,
    PlanStepStatus,
    ToolDefinition,
    ToolResult,
)
from core.intent_router import IntentRouter
from core.task_runtime import TaskRuntimeModule
from core.tools import Tool
from core.transliteration import canonicalize_command, strip_wake_remnant


# --------------------------------------------------------------------------
# Transcript cleanup
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Upyasta Upnro", "Upnro"),
        ("Ugyasta open chrome then open camera", "open chrome then open camera"),
        ("hey asta open chrome", "open chrome"),
        ("Hello Aster Open Chrome", "Open Chrome"),
        ("ओप्यास्टा ओपन क्रो", "ओपन क्रो"),
        ("उज्ज्यास्ट ओपन क्रोम देन ओपन कैमरा", "ओपन क्रोम देन ओपन कैमरा"),
        # Never strip ordinary words or a bare wake word.
        ("pasta recipe please", "pasta recipe please"),
        ("पास्टा बनाओ", "पास्टा बनाओ"),
        ("रास्ता बताओ", "रास्ता बताओ"),
        ("Asta", "Asta"),
        ("hey asta", "hey asta"),
    ],
)
def test_strip_wake_remnant(text, expected):
    assert strip_wake_remnant(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ओपन क्रोम", "open chrome"),
        ("क्रोम खोलो", "open chrome"),
        ("क्रोम को बंद करो", "close chrome"),
        ("chrome kholo aur camera kholo", "open chrome then open camera"),
        ("spotify band karo", "close spotify"),
        ("गाना बजाओ", "play music"),
        ("आज मौसम कैसा है", "आज मौसम कैसा है"),
        ("what is the capital of france", "what is the capital of france"),
    ],
)
def test_canonicalize_command(text, expected):
    assert canonicalize_command(text) == expected


@pytest.mark.parametrize(
    "text, entities",
    [
        ("ओप्यास्टा ओपन क्रो", {"action": "open", "target": "chrome"}),
        ("क्रोम खोलो", {"action": "open", "target": "chrome"}),
        (
            "Ugyasta open chrome then open camera",
            {"commands": [
                {"action": "open", "target": "chrome"},
                {"action": "open", "target": "camera"},
            ]},
        ),
    ],
)
def test_intent_router_understands_noisy_voice_commands(text, entities):
    result = IntentRouter().analyze(text)
    assert result.intent is IntentType.COMMAND
    assert result.classifier == "rules"
    assert result.entities == entities


@pytest.mark.parametrize(
    "english, hindi, expected",
    [
        # English decode clipped the target; Hindi decode kept it.
        ("Open", "ओपन क्रोम", "open chrome"),
        # Wake word merged and English decode garbled.
        ("Upyasta Upnro", "ओप्यास्टा ओपन क्रो", "open chrome"),
        ("", "ओपन क्रोम", "open chrome"),
        # Equal evidence keeps the English decode as spoken.
        ("Open chrome", "ओपन क्रोम", "Open chrome"),
    ],
)
def test_choose_transcript_recovers_devanagari_english(english, hindi, expected):
    from voice.language_choice import choose_transcript

    assert choose_transcript(english, hindi) == ("en", expected)


# --------------------------------------------------------------------------
# System 1 planning gate
# --------------------------------------------------------------------------
class FakeOpenTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.open",
            description="Open or run something for tests.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["open", "run"]},
        )

    def execute(self, request):
        raise AssertionError("planner tests must not execute tools")


class RecordingBrain:
    enabled = True

    def __init__(self):
        self.calls = 0

    def plan(self, goal, *, intent):
        self.calls += 1
        return AgentPlanProposal(
            goal_summary="Do it.",
            success_conditions=("Done.",),
            rationale="Test.",
            uncertainty=0.1,
            steps=({"action": "open", "target": "calculator"},),
        )

    @staticmethod
    def task_metadata(proposal):
        return {"agent_mode": "cognitive_v1"}


def _planner(brain, decision_engine=None):
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    return Planner(
        kernel.tool_registry,
        agent_brain=brain,
        decision_engine=decision_engine,
    )


def _intent(entities, *, classifier="rules", confidence=0.98):
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=confidence,
        normalized_text="test",
        entities=entities,
        requires_tools=True,
        classifier=classifier,
    )


def test_rules_command_skips_llm_planner(monkeypatch):
    monkeypatch.delenv("ASTA_AGENT_PLANNING", raising=False)
    brain = RecordingBrain()
    plan = _planner(brain).plan(
        "open calculator",
        intent=_intent({"action": "open", "target": "calculator"}),
    )
    assert brain.calls == 0
    assert plan.metadata["planner"] == "deterministic"
    assert [step.description for step in plan.steps] == ["open calculator"]


def test_compound_rules_command_skips_llm_planner(monkeypatch):
    monkeypatch.delenv("ASTA_AGENT_PLANNING", raising=False)
    brain = RecordingBrain()
    plan = _planner(brain).plan(
        "open chrome then open camera",
        intent=_intent({"commands": [
            {"action": "open", "target": "chrome"},
            {"action": "open", "target": "camera"},
        ]}),
    )
    assert brain.calls == 0
    assert len(plan.steps) == 2


def test_llm_planner_still_used_when_forced_or_needed(monkeypatch):
    brain = RecordingBrain()
    planner = _planner(brain)

    monkeypatch.setenv("ASTA_AGENT_PLANNING", "always")
    planner.plan("open calculator", intent=_intent({"action": "open", "target": "calculator"}))
    assert brain.calls == 1

    monkeypatch.delenv("ASTA_AGENT_PLANNING")
    # Low confidence and non-direct actions go to the LLM planner.
    planner.plan(
        "open calculator",
        intent=_intent({"action": "open", "target": "calculator"}, confidence=0.6),
    )
    planner.plan("run backup", intent=_intent({"action": "run", "target": "backup"}))
    assert brain.calls == 3


class FakeLaya:
    name = "laya"

    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def decide_action(self, text, **kwargs):
        self.calls += 1
        return self.decision


def test_laya_gate_decides_whether_llm_planning_is_needed(monkeypatch):
    monkeypatch.delenv("ASTA_AGENT_PLANNING", raising=False)
    intent = _intent({"action": "run", "target": "backup"})

    simple = FakeLaya(ActionDecision(
        action=ActionType.SYSTEM, confidence=0.93, addressed=0.9,
        command_complete=True, compound=False, source="laya",
    ))
    brain = RecordingBrain()
    _planner(brain, simple).plan("run backup", intent=intent)
    assert simple.calls == 1
    assert brain.calls == 0

    complex_task = FakeLaya(ActionDecision(
        action=ActionType.TASK, confidence=0.9, addressed=0.9,
        command_complete=True, compound=True, source="laya",
    ))
    brain = RecordingBrain()
    _planner(brain, complex_task).plan("run backup", intent=intent)
    assert brain.calls == 1


# --------------------------------------------------------------------------
# Post-action LLM decision
# --------------------------------------------------------------------------
def _task(goal, statuses):
    steps = [
        SimpleNamespace(id=f"step-{i}", status=status)
        for i, status in enumerate(statuses, start=1)
    ]
    return SimpleNamespace(goal=goal, plan=SimpleNamespace(steps=steps))


def _ok(tool="system.open_application"):
    return ToolResult(success=True, tool=tool, output={})


def test_post_action_decision_skipped_while_plan_continues(monkeypatch):
    monkeypatch.delenv("ASTA_AGENT_POST_ACTION", raising=False)
    task = _task("open chrome then open camera", [PlanStepStatus.RUNNING, PlanStepStatus.PENDING])
    assert TaskRuntimeModule._post_action_skip_reason(task, _ok(), "step-1", evidence={})


def test_post_action_decision_skipped_for_verified_final_step(monkeypatch):
    monkeypatch.delenv("ASTA_AGENT_POST_ACTION", raising=False)
    task = _task("open chrome", [PlanStepStatus.RUNNING])
    verified = {"verification": {"status": "verified"}}
    assert TaskRuntimeModule._post_action_skip_reason(task, _ok(), "step-1", evidence=verified)
    # Unverified final steps and explicit verification goals still reason.
    assert TaskRuntimeModule._post_action_skip_reason(task, _ok(), "step-1", evidence={}) is None
    explicit = _task("open chrome and verify it is open", [PlanStepStatus.RUNNING])
    assert TaskRuntimeModule._post_action_skip_reason(explicit, _ok(), "step-1", evidence=verified) is None


def test_post_action_decision_kept_for_vision_and_when_forced(monkeypatch):
    task = _task("look at the screen", [PlanStepStatus.RUNNING, PlanStepStatus.PENDING])
    assert TaskRuntimeModule._post_action_skip_reason(
        task, _ok("vision.inspect"), "step-1", evidence={}
    ) is None
    monkeypatch.setenv("ASTA_AGENT_POST_ACTION", "always")
    assert TaskRuntimeModule._post_action_skip_reason(task, _ok(), "step-1", evidence={}) is None


def test_truncated_post_action_decision_is_repaired():
    # Real output cut off by the 128-token budget, with "none" placeholders.
    response = """{
"goal_satisfied": true,
"needs_observation": false,
"needs_user": false,
"rationale": "Chrome is open and no error occurred.",
"confidence": 1.0,
"uncertainty": 0.0,
"next_action": {
"action": "none",
"tool": "none",
"target": "none"
},
"belief_updates": []"""
    decision = AgentBrain._parse_decision(response)
    assert decision.goal_satisfied is True
    assert decision.next_action is None
    assert decision.rationale.startswith("Chrome is open")


# --------------------------------------------------------------------------
# Speech output
# --------------------------------------------------------------------------
def test_streamed_acronym_is_not_spoken_as_its_own_sentence():
    from ai.llama_cpp_engine import LlamaCppEngine

    buffer, sentences = "", []
    for token in ["A.", "S.", "T.", "A.", " remembers", " Go.", " How", " can I help?"]:
        buffer += token
        while True:
            buffer, sentence = LlamaCppEngine._emit_sentence_chunks(buffer)
            if sentence is None:
                break
            sentences.append(sentence)
    assert sentences == ["A.S.T.A. remembers Go.", "How can I help?"]


def test_kokoro_replays_short_phrases_from_cache():
    from speech.kokoro_engine import KokoroEngine

    calls = []

    def pipeline(text, **kwargs):
        calls.append(text)
        yield None, None, np.full(240, 0.2, dtype=np.float32)

    engine = object.__new__(KokoroEngine)
    engine.speed = 1.0
    engine.first_clause_enabled = False
    engine.cache_max_chars = 80
    engine.cache_size = 2
    engine._audio_cache = OrderedDict()
    engine._cache_lock = threading.Lock()
    engine._pipeline_for = lambda text: (pipeline, "am_michael")

    first = list(engine.iter_audio("Opened chrome."))
    second = list(engine.iter_audio("Opened chrome."))
    assert len(calls) == 1
    assert np.array_equal(first[0], second[0])

    long_text = "x" * 100
    list(engine.iter_audio(long_text))
    list(engine.iter_audio(long_text))
    assert calls.count(long_text) == 2


def test_laya_recovers_application_command_from_unusual_phrasing():
    from ai.ai_module import AIModule

    class Laya:
        name = "laya"

        def __init__(self):
            self.applications = None

        def decide_action(self, text, *, applications=(), media_providers=()):
            self.applications = [app.name for app in applications]
            return ActionDecision(
                action=ActionType.OPEN_APP, confidence=0.91, addressed=0.95,
                arguments={"target_app": "Google Chrome"},
                command_complete=True, compound=False, source="laya",
                latency_ms=12.0,
            )

    class Apps:
        def discover(self, query, limit=8):
            return [SimpleNamespace(name="Google Chrome")] if "chrome" in query else []

    events = []
    module = object.__new__(AIModule)
    module.event_bus = SimpleNamespace(emit=lambda *a, **k: events.append(a))
    laya = Laya()
    module.kernel = SimpleNamespace(
        decision_engine=laya,
        application_manager=Apps(),
        media_manager=None,
    )
    handled = []
    module._handle_command_intent = handled.append

    hint = IntentResult(
        intent=IntentType.UNKNOWN, confidence=0.2,
        normalized_text="fire up chrome for me", classifier="rules",
    )
    assert module._run_system1_decision("fire up chrome for me", intent_hint=hint) is True
    assert laya.applications == ["Google Chrome"]
    assert handled[0].entities == {"action": "open", "target": "Google Chrome"}
    assert handled[0].classifier == "laya_system1"


# --------------------------------------------------------------------------
# Second voice session fixes
# --------------------------------------------------------------------------
def test_choose_transcript_fuzzy_matches_misspelled_app_name():
    from voice.language_choice import choose_transcript

    assert choose_transcript("Open ground", "ओपन क्रम") == ("en", "open chrome")
    # Common Hindi words are never fuzzy-matched to app names.
    assert choose_transcript("volume kam karo", "वॉल्यूम कम करो")[0] == "hi"


def test_clipped_search_query_does_not_type_filler():
    result = IntentRouter().analyze("search for")
    assert result.entities.get("action") != "search"
    compound = IntentRouter().analyze("open chrome and search for")
    commands = compound.entities.get("commands") or [compound.entities]
    assert all(command.get("query") != "for" for command in commands)


def test_user_directed_marker_only_confirms_keyboard_steps(monkeypatch):
    from core.tools.module import ToolRuntimeModule

    monkeypatch.delenv("ASTA_TRUST_USER_DIRECTED", raising=False)
    typing = SimpleNamespace(tool="computer.type_text", metadata={"user_directed": True})
    shell = SimpleNamespace(tool="system.run_command", metadata={"user_directed": True})
    unmarked = SimpleNamespace(tool="computer.type_text", metadata={})
    assert ToolRuntimeModule._is_user_directed(typing) is True
    assert ToolRuntimeModule._is_user_directed(shell) is False
    assert ToolRuntimeModule._is_user_directed(unmarked) is False
    monkeypatch.setenv("ASTA_TRUST_USER_DIRECTED", "0")
    assert ToolRuntimeModule._is_user_directed(typing) is False


def test_deterministic_search_plan_marks_typing_user_directed():
    planner = Planner(Kernel().tool_registry)
    commands = planner._expand_search_commands(
        [{"action": "open", "target": "chrome"},
         {"action": "search", "query": "weather"}],
        user_directed=True,
    )
    marked = {c["action"] for c in commands if c.get("user_directed")}
    # Without a hotkey tool registered, the vision locate/click path is used.
    assert marked == {"type_text", "keypress"}
    llm = planner._expand_search_commands(
        [{"action": "search", "query": "weather", "target": "chrome"}],
    )
    assert not any(c.get("user_directed") for c in llm)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("i just said yes", (True, False)),
        ("yes please do it", (True, False)),
        ("हाँ", (True, False)),
        ("no don't", (False, True)),
        ("okay", (False, False)),
        ("yes no", (False, False)),
        ("open chrome and search for the weather in delhi today please now", (False, False)),
    ],
)
def test_short_approval_classification(text, expected):
    from ai.ai_module import AIModule

    assert AIModule._classify_short_approval(text) == expected


def test_quiet_utterance_retry_boosts_short_audio():
    from voice.voice_module import VoiceModule

    seen = []
    module = object.__new__(VoiceModule)
    module.recognition = SimpleNamespace(
        transcribe=lambda audio: seen.append(audio) or "yes"
    )
    audio = np.full(8000, 0.1, dtype=np.float32)
    assert module._retry_quiet_utterance(audio) == "yes"
    assert seen[0].size == 8000 + 4800
    assert np.isclose(np.max(seen[0]), 0.6)
    # Long utterances are not retried.
    assert module._retry_quiet_utterance(np.full(16000 * 4, 0.1, dtype=np.float32)) == ""


def test_dangling_clause_keeps_first_command():
    result = IntentRouter().analyze("open chrome and search for")
    assert result.entities == {"action": "open", "target": "chrome"}


# --------------------------------------------------------------------------
# Voice fixes 2: aliases, implicit splits, turn holding, cleanup
# --------------------------------------------------------------------------


def test_misheard_browser_name_and_implicit_second_command():
    from core.transliteration import apply_target_aliases, split_implicit_commands

    assert apply_target_aliases("open ground") == "open chrome"
    assert apply_target_aliases("ground beef recipe") == "ground beef recipe"
    assert split_implicit_commands("open chrome search for weather") == (
        "open chrome then search for weather"
    )
    assert split_implicit_commands("open chrome, search for weather") == (
        "open chrome, search for weather"
    )
    result = IntentRouter().analyze("Open ground search for weather")
    actions = [c.get("action") for c in result.entities.get("commands", [])]
    assert actions == ["open", "search"]
    assert result.entities["commands"][0]["target"] == "chrome"


def test_user_voice_aliases_file(tmp_path, monkeypatch):
    import core.transliteration as tr

    path = tmp_path / "aliases.json"
    path.write_text('{"note pad plus": "notepad++"}', encoding="utf-8")
    monkeypatch.setenv("ASTA_VOICE_ALIASES", str(path))
    monkeypatch.setattr(tr, "_alias_cache", None)
    assert tr.apply_target_aliases("open note pad plus") == "open notepad++"


def test_ends_unfinished():
    from core.transliteration import ends_unfinished

    assert ends_unfinished("open")
    assert ends_unfinished("open chrome and")
    assert ends_unfinished("search for")
    assert not ends_unfinished("open chrome")
    assert not ends_unfinished("")


class _FakeDetector:
    threshold = 0.5

    def __init__(self, probability):
        self.probability = probability

    def completion_probability(self, audio):
        return self.probability

    def is_complete(self, audio):
        return self.probability >= self.threshold


def test_turn_tracker_short_turn_needs_higher_confidence():
    from voice.smart_turn import TurnEndTracker

    tracker = TurnEndTracker(
        _FakeDetector(0.7), 8000, short_speech_samples=14400, short_threshold=0.85
    )
    tracker.speech_samples = 8000
    assert not tracker._sounds_complete(np.zeros(8000, dtype=np.float32))
    tracker.speech_samples = 20000
    assert tracker._sounds_complete(np.zeros(20000, dtype=np.float32))


def test_turn_tracker_hold_waits_for_full_silence():
    from voice.smart_turn import TurnEndTracker

    tracker = TurnEndTracker(_FakeDetector(0.99), 8000, hold=lambda: True)
    audio = np.zeros(20000, dtype=np.float32)
    assert not tracker.update(started=False, ended=True, chunk_samples=20000, get_audio=lambda: audio)
    assert tracker.pending
    assert tracker.update(started=False, ended=False, chunk_samples=8000, get_audio=lambda: audio)


def test_speech_strips_emoji():
    from speech.speech_module import SpeechModule

    cleaned = SpeechModule._prepare_for_speech("Sure thing! 😊👍")
    assert "😊" not in cleaned and "👍" not in cleaned
    assert "Sure thing" in cleaned


def test_missing_app_failure_asks_to_repeat():
    from ai.ai_module import AIModule

    message = AIModule._format_tool_failure(
        ToolResult(
            success=False,
            tool="system.open_application",
            output={"target": "ground"},
            error="No installed application matched 'ground'.",
        )
    )
    assert "couldn't find an app called ground" in message


def test_completion_message_overrides_generic_success():
    from ai.ai_module import AIModule

    message = AIModule._format_tool_success(
        ToolResult(
            success=True,
            tool="computer.keypress",
            metadata={"completion_message": "Searched for weather."},
        )
    )
    assert message == "Searched for weather."


def test_vision_idle_unload_stops_owned_server(monkeypatch):
    from vision.lfm2_5_vl_engine import LFM25VLEngine

    stopped = []
    engine = LFM25VLEngine.__new__(LFM25VLEngine)
    engine.server_manager = SimpleNamespace(owned=True, stop=lambda: stopped.append(1))
    engine._idle_timer = None
    monkeypatch.setenv("ASTA_VISION_IDLE_UNLOAD_SECONDS", "0.05")
    engine._inspect = lambda *a, **k: "ok"
    assert engine.inspect() == "ok"
    engine._idle_timer.join(1)
    assert stopped == [1]


def test_spoken_transition_between_commands_is_split():
    for text in (
        "Okay, open chrome and once you are there, search for Hanchalisa",
        "open chrome, when it opens search for cats",
        "open chrome and after that search for cats",
    ):
        commands = IntentRouter().analyze(text).entities.get("commands", [])
        assert [c.get("action") for c in commands] == ["open", "search"], text
        assert commands[0]["target"] == "chrome"
    # A transition-like phrase inside a query is left alone.
    result = IntentRouter().analyze("search for when you are there movie")
    assert result.entities["query"] == "when you are there movie"


def test_hindi_decode_repairs_garbled_search_query():
    from voice.language_choice import choose_transcript

    language, text = choose_transcript(
        "Open chrome and once you are there search for human challenges",
        "ओपन क्रोम एंड वन सी आर देर सर्च फॉर हनुमान चलीसा",
    )
    assert language == "en"
    assert text.endswith("search for hanuman chalisa")
    # English queries without Hindi names stay as decoded.
    assert choose_transcript("Search for weather", "सर्च फॉर वेदर")[1] == "Search for weather"
    assert choose_transcript(
        "Search for quantum computing", "सर्च फॉर क्वांटम कंप्यूटिंग"
    )[1] == "Search for quantum computing"


def test_verb_typos_are_fixed():
    result = IntentRouter().analyze("Serch for cats")
    assert result.entities == {"action": "search", "query": "cats"}


def test_gpu_share_releases_only_idle_users(monkeypatch):
    import time as _time
    from core import gpu_share

    released = []
    monkeypatch.setenv("ASTA_GPU_RELEASE_IDLE_SECONDS", "5")
    gpu_share.register("busy", lambda: released.append("busy"), lambda: None)
    gpu_share.register("recent", lambda: released.append("recent"), lambda: _time.monotonic())
    gpu_share.register("idle", lambda: released.append("idle"), lambda: _time.monotonic() - 60)
    try:
        assert gpu_share.release_idle() == ["idle"]
        assert released == ["idle"]
    finally:
        for name in ("busy", "recent", "idle"):
            gpu_share.unregister(name)


@pytest.mark.parametrize(
    "english, hindi",
    [
        ("Okay, search for human challenge so on chrome", "सर्च फॉर हनुमान चली सॉन क्रोम"),
        ("Okay, search for machine Chrome", "ओके सर्च फॉर हनुमान चलीस ऑन क्रोम"),
        ("Serch for human challeng", "सर्च फॉर हनुमान चली साहब"),
        ("Search for human challenges", "सर्च फॉर हनुमान चालीसा"),
        ("Sermanchali Son", "सर्च फॉर हनुमान चले सौ उन क्रोम"),
    ],
)
def test_hindi_names_survive_different_phrasings(english, hindi):
    from voice.language_choice import choose_transcript

    _, text = choose_transcript(english, hindi)
    assert IntentRouter().analyze(text).entities["query"] == "hanuman chalisa"


def test_partial_transcripts_are_not_repaired():
    from voice.language_choice import choose_transcript

    _, text = choose_transcript(
        "search for human challe", "सर्च फॉर हनुमान चली", repair=False
    )
    assert text == "search for human challe"


def test_browser_search_uses_address_bar_shortcut(monkeypatch):
    monkeypatch.delenv("ASTA_BROWSER_ADDRESS_BAR", raising=False)
    assert Planner._is_browser("Google Chrome")
    assert Planner._is_browser("msedge.exe")
    assert not Planner._is_browser("spotify")
    steps = Planner._expand_search_commands(
        SimpleNamespace(
            application_manager=None,
            _is_browser=Planner._is_browser,
            _has_tool=lambda name: name == "computer.hotkey",
        ),
        [{"action": "search", "query": "hanuman chalisa", "target": "chrome"}],
        user_directed=True,
    )
    tools = [step.get("tool") or step["action"] for step in steps]
    assert tools == ["open", "computer.wait", "computer.hotkey", "computer.type_text", "computer.keypress"]
    assert steps[2]["keys"] == ["ctrl", "l"] and steps[2]["user_directed"] is True


def test_only_address_bar_hotkey_is_pre_approved():
    from core.tools.module import ToolRuntimeModule

    def request(keys):
        return SimpleNamespace(
            tool="computer.hotkey", arguments={"keys": keys}, metadata={"user_directed": True}
        )

    assert ToolRuntimeModule._is_user_directed(request(["ctrl", "l"]))
    assert not ToolRuntimeModule._is_user_directed(request(["alt", "f4"]))


@pytest.mark.parametrize(
    "english, hindi",
    [
        ("", "सर्च पर हनुमान चलेशन क्रोम"),
        ("Shurma salis on chrome", "सच पर हनुमान चलीस ऑन क्रोम"),
    ],
)
def test_search_heard_only_in_hindi_stream(english, hindi):
    from voice.language_choice import choose_transcript

    language, text = choose_transcript(english, hindi)
    assert language == "en"
    entities = IntentRouter().analyze(text).entities
    assert entities == {"action": "search", "query": "hanuman chalisa", "target": "chrome"}


def test_hindi_sentences_with_sach_stay_hindi():
    from voice.language_choice import choose_transcript

    assert choose_transcript("", "सच में क्या हुआ") == ("hi", "सच में क्या हुआ")


def test_hotkey_plan_step_with_key_list_builds_request():
    from core.tools.computer import ComputerHotkeyTool

    kernel = Kernel()
    kernel.register_tool(ComputerHotkeyTool(SimpleNamespace()))
    tasks = TaskRuntimeModule(kernel)
    task = SimpleNamespace(
        id="task-1", goal="search", metadata={},
        plan=SimpleNamespace(metadata={"planner": "deterministic"}),
    )
    step = SimpleNamespace(
        id="step-3",
        description="hotkey",
        metadata={"action": "hotkey", "tool": "computer.hotkey", "keys": ["ctrl", "l"], "user_directed": True},
    )
    request = tasks.build_plan_request(task, step)
    assert request.tool == "computer.hotkey"
    assert request.arguments["keys"] == ["ctrl", "l"]
    assert request.metadata["user_directed"] is True


def test_clipped_command_starts_are_recovered():
    from voice.language_choice import choose_transcript

    result = IntentRouter().analyze("Once you are there search for hanuman chalisa")
    assert result.entities == {"action": "search", "query": "hanuman chalisa"}
    _, text = choose_transcript("", "पर हनुमान चले से वन क्रो")
    assert IntentRouter().analyze(text).entities == {
        "action": "search", "query": "hanuman chalisa", "target": "chrome",
    }
    # Ordinary Hindi starting with "पर" (but) is not a search.
    assert choose_transcript("", "पर वो क्रोम")[0] == "hi"
    assert choose_transcript("", "पर मुझे नहीं पता")[0] == "hi"


def test_fake_tool_narration_is_not_spoken():
    from speech.speech_module import SpeechModule

    spoken = SpeechModule._prepare_for_speech(
        '[Using notes.search_notes for "Hanuman Chalisa"] Found: Hanuman Chalisa.'
    )
    assert "Using" not in spoken and "Found" in spoken


def test_browser_search_launches_results_page_directly(monkeypatch, tmp_path):
    from core.tools.browser import BrowserSearchTool, chromium_profile, search_url
    from core.tools import ComputerHotkeyTool

    kernel = Kernel()
    kernel.register_tool(BrowserSearchTool(popen=lambda *a, **k: None))
    kernel.register_tool(ComputerHotkeyTool(SimpleNamespace()))
    planner = Planner(kernel.tool_registry)
    steps = planner._expand_search_commands(
        [{"action": "open", "target": "chrome"},
         {"action": "search", "query": "hanuman chalisa"}],
        user_directed=True,
    )
    assert [s["action"] for s in steps] == ["web_search"]
    assert steps[0]["browser"] == "chrome"
    assert steps[0]["completion_message"] == "Searched for hanuman chalisa."
    assert search_url("hanuman chalisa") == "https://www.google.com/search?q=hanuman+chalisa"

    # Last used Chrome profile skips "Who's using Chrome?".
    data = tmp_path / "Google" / "Chrome" / "User Data"
    data.mkdir(parents=True)
    (data / "Local State").write_text('{"profile": {"last_used": "Profile 2"}}', encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("ASTA_BROWSER_PROFILE", raising=False)
    assert chromium_profile("chrome") == "Profile 2"
    monkeypatch.setenv("ASTA_BROWSER_PROFILE", "Default")
    assert chromium_profile("chrome") == "Default"


def test_browser_search_tool_builds_profile_command(monkeypatch, tmp_path):
    import core.tools.browser as browser

    launched = []
    monkeypatch.setattr(browser, "_find_executable", lambda key: "C:/chrome.exe")
    monkeypatch.setenv("ASTA_BROWSER_PROFILE", "Profile 1")
    tool = browser.BrowserSearchTool(popen=lambda cmd, **k: launched.append(cmd))
    request = SimpleNamespace(arguments={"query": "cats", "browser": "Google Chrome"}, request_id="r1")
    result = tool.execute(request)
    assert result.success
    assert launched == [["C:/chrome.exe", "--profile-directory=Profile 1",
                         "https://www.google.com/search?q=cats"]]


def test_search_without_app_is_a_web_search(monkeypatch):
    from core.tools.browser import BrowserSearchTool

    monkeypatch.delenv("ASTA_DEFAULT_BROWSER", raising=False)
    kernel = Kernel()
    kernel.register_tool(BrowserSearchTool(popen=lambda *a, **k: None))
    planner = Planner(kernel.tool_registry)
    steps = planner._expand_search_commands(
        [{"action": "search", "query": "new songs"}], user_directed=True
    )
    assert steps == [{
        "action": "web_search", "tool": "browser.search", "query": "new songs",
        "browser": "chrome", "completion_message": "Searched for new songs.",
    }]


def test_full_english_sentence_beats_devanagari_guess():
    from voice.language_choice import choose_transcript

    assert choose_transcript(
        "Search for model Router on Chrome", "सर्च फॉर मॉडल राउटर ऑन क्रोम"
    )[1] == "Search for model Router on Chrome"
    assert choose_transcript("Search for human challeng", "सर्च फॉर हनुमान चले सॉन क्रोम")[1] == (
        "Search for hanuman chalisa on chrome"
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What is two plus two", "That's 4."),
        ("what is 12 times 7?", "That's 84."),
        ("how much is one hundred and five minus five", "That's 100."),
        ("what is 10 divided by 4", "That's 2.5."),
        ("what is the capital of france", None),
        ("search for one plus", None),
        ("what is 2 to the power of 1000", None),
    ],
)
def test_quick_math_answers_without_llm(text, expected):
    from core.quick_answers import quick_math

    assert quick_math(text) == expected


def test_adopted_orphan_vision_server_is_killed(monkeypatch):
    import vision.vision_server_manager as vsm

    killed = []
    monkeypatch.setattr(vsm.os, "name", "nt")
    monkeypatch.setattr(vsm, "_listening_pid", lambda port: 4242)
    monkeypatch.setattr(vsm, "_process_name", lambda pid: "llama-server.exe")
    monkeypatch.setattr(vsm, "_kill_pid", lambda pid: killed.append(pid))
    manager = vsm.VisionServerManager.__new__(vsm.VisionServerManager)
    manager.base_url = "http://127.0.0.1:8090/v1"
    manager.process = None
    manager.owned = False
    manager._adopted_pid = None
    manager.preload = False
    manager._server_is_ready = lambda: True
    assert manager.release_orphan() is True
    assert killed == [4242] and manager.owned is False


@pytest.mark.parametrize(
    ("english", "hindi", "expected"),
    [
        ("Photoshop", "फोटोशॉप", "Photoshop"),
        ("For Photoshop", "फॉर फोटोशॉप", "For Photoshop"),
    ],
)
def test_choose_transcript_keeps_real_english_over_fuzzy_hindi(english, hindi, expected):
    from voice.language_choice import choose_transcript

    language, text = choose_transcript(english, hindi)
    assert language == "en"
    assert "photos" != text.lower()
    assert text.lower().endswith("photoshop")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("play the last song I was listening to", {"action": "media", "operation": "play"}),
        ("play that song", {"action": "media", "operation": "play"}),
        ("can you search song for me", {"action": "search", "query": "song"}),
        (
            "for smartphones on chrome",
            {"action": "search", "query": "smartphones", "target": "chrome"},
        ),
        ("open notes", {"action": "open", "target": "sticky notes"}),
    ],
)
def test_router_handles_test_log_phrasings(text, expected):
    entities = IntentRouter().analyze(text).entities
    for key, value in expected.items():
        assert entities.get(key) == value
    if expected.get("action") == "media":
        assert not entities.get("query")


@pytest.mark.parametrize(
    "text",
    ["Can you see the search field?", "what's on my screen", "look at my screen"],
)
def test_screen_questions_route_to_vision_inspect(text):
    entities = IntentRouter().analyze(text).entities
    assert entities["action"] == "inspect"
    assert entities["prompt"].startswith(IntentRouter.SCREEN_QUESTION_PREFIX)


@pytest.mark.parametrize("text", ["can you find a good restaurant", "can you read me a story"])
def test_non_screen_questions_do_not_use_vision(text):
    assert IntentRouter().analyze(text).entities.get("action") != "inspect"


def test_quick_math_square_roots():
    from core.quick_answers import quick_math

    assert quick_math("what is the square root of 144") == "That's 12."
    assert quick_math("cube root of 27") == "That's 3."
    assert quick_math("square root of apples") is None


def test_open_failure_names_the_app():
    from core.applications import ApplicationResolutionError
    from core.contracts.tools import ToolRequest
    from core.tools.system import OpenApplicationTool

    tool = OpenApplicationTool()

    def fail(_target):
        raise ApplicationResolutionError("Ambiguous application 'notes'.")

    tool._browser_profile_command = lambda _target: None
    tool.resolve_target = fail
    result = tool.execute(
        ToolRequest(
            tool="system.open_application",
            arguments={"target": "notes"},
            request_id="t1",
        )
    )
    assert not result.success
    assert result.output["target"] == "notes"


def test_unconfigured_spotify_play_uses_the_app_ui(monkeypatch):
    from core import Planner as _Planner

    monkeypatch.delenv("ASTA_SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("ASTA_MEDIA_API_SETUP", raising=False)
    kernel = Kernel()
    planner = _Planner(
        kernel.tool_registry,
        media_manager=kernel.media_manager,
        application_manager=kernel.application_manager,
    )
    commands = planner._expand_media_commands(
        [{"action": "media", "operation": "play", "query": "hanuman chalisa", "provider": "spotify"}]
    )
    tools = [c.get("tool") or c.get("action") for c in commands]
    # Spotify's deep link opens the results page; no search box typing.
    assert commands[0] == {"action": "open", "target": "spotify:search:hanuman%20chalisa"}
    assert "media" not in tools and "computer.type_text" not in tools
    assert any(c.get("clicks") == 2 for c in commands)
    assert tools[-1] == "vision.inspect"


def test_media_gui_typing_is_user_directed_and_hud_steps_aside(monkeypatch):
    from core import Planner as _Planner
    from hud.hud_module import HUDModule

    kernel = Kernel()
    planner = _Planner(kernel.tool_registry)
    steps = planner._interactive_media_play_steps(
        query="song", application="Music", user_directed=True
    )
    typed = [s for s in steps if s["tool"] in {"computer.type_text", "computer.keypress"}]
    assert typed and all(s.get("user_directed") for s in typed)

    sent = []
    hud = HUDModule.__new__(HUDModule)
    hud.transport = SimpleNamespace(publish_window=sent.append)
    hud.set_state = lambda **kw: None
    monkeypatch.setenv("ASTA_HUD_HIDE_SETTLE", "0")
    hud.on_tool_request(SimpleNamespace(tool="vision.locate"))
    hud.on_tool_request(SimpleNamespace(tool="computer.click"))
    hud._restore_window()
    assert sent == ["minimize", "restore"]


def test_setup_error_is_not_relabelled_by_system_one():
    from core.autonomy.diagnosis import DiagnosisCategory, DiagnosisEngine, FailureDiagnosis

    class Guess:
        def diagnose_failure(self, payload):
            fallback = DiagnosisEngine._fallback(payload)
            fallback.category = DiagnosisCategory.UNKNOWN
            return fallback

    result = ToolResult(
        success=False,
        tool="media.control",
        error="Spotify playback requires one-time setup. Set ASTA_SPOTIFY_CLIENT_ID.",
    )
    task = SimpleNamespace(id="t", goal="play x on spotify", evidence=[])
    diagnosis = DiagnosisEngine(Guess()).diagnose(task, result)
    assert isinstance(diagnosis, FailureDiagnosis)
    assert diagnosis.category is DiagnosisCategory.SETUP_REQUIRED


def test_filler_words_and_launch_it_reference():
    router = IntentRouter()
    assert router.analyze("Okay, can you uh open Chrome").entities == {
        "action": "open", "target": "chrome"
    }
    assert router.analyze("Launch it now").entities == {"action": "open", "target": "it"}

    from core.applications import ApplicationManager

    manager = ApplicationManager()
    manager.last_mentioned_application = "Google Chrome"
    assert manager.resolve_reference("it") == "Google Chrome"


class _Runner:
    def __init__(self, found, titles):
        self.found = found
        self.titles = list(titles)
        self.calls = []

    def __call__(self, script, env, timeout):
        import json as _json

        self.calls.append(env)
        if "ASTA_UI_TOKENS" in env:
            return _json.dumps(self.found)
        return self.titles.pop(0) if self.titles else ""


def _ui_request(query="hanuman chalisa"):
    from core.contracts.tools import ToolRequest

    return ToolRequest(
        tool="media.ui_play",
        arguments={"query": query, "application": "Spotify", "process": "Spotify"},
        request_id="r",
    )


def test_ui_play_invokes_named_button_and_verifies_title():
    from core.tools.ui_play import UIPlayTool

    runner = _Runner(
        {"invoked": True, "name": "Play Shree Hanuman Chalisa by Hariharan"},
        ["Spotify Premium", "Spotify Premium", "Hariharan - Shree Hanuman Chalisa"],
    )
    result = UIPlayTool(runner=runner, sleep=lambda _s: None).execute(_ui_request())
    assert result.success
    assert result.output["method"] == "ui_automation"
    assert runner.calls[1]["ASTA_UI_TOKENS"] == "hanuman|chalisa"
    assert result.output["message"] == "Playing Shree Hanuman Chalisa by Hariharan."


def test_ui_play_falls_back_to_vision_double_click():
    from core.tools.ui_play import UIPlayTool

    clicks = []
    locate = SimpleNamespace(
        execute=lambda req: ToolResult(
            success=True, tool="vision.locate", output={"screen_center": {"x": 10, "y": 20}}
        )
    )
    controller = SimpleNamespace(click=lambda **kw: clicks.append(kw))
    runner = _Runner({"invoked": False, "error": "no_match"}, ["Spotify Free", "Artist - Hanuman Chalisa"])
    result = UIPlayTool(
        runner=runner, locate_tool=locate, controller=controller, sleep=lambda _s: None
    ).execute(_ui_request())
    assert result.success and result.output["method"] == "vision_double_click"
    assert clicks[0]["clicks"] == 2 and clicks[0]["x"] == 10


def test_ui_play_reports_failure_when_title_never_changes():
    from core.tools.ui_play import UIPlayTool

    runner = _Runner({"invoked": True, "name": "Play x"}, ["Spotify Free"] * 8)
    result = UIPlayTool(runner=runner, sleep=lambda _s: None).execute(_ui_request())
    assert not result.success and result.output["verified"] is False


def test_spotify_plan_uses_ui_play_tool(monkeypatch):
    from core import Planner as _Planner
    from core.tools import UIPlayTool

    monkeypatch.delenv("ASTA_SPOTIFY_CLIENT_ID", raising=False)
    kernel = Kernel()
    kernel.register_tool(UIPlayTool(runner=lambda *a: ""))
    planner = _Planner(
        kernel.tool_registry,
        media_manager=kernel.media_manager,
        application_manager=kernel.application_manager,
    )
    commands = planner._expand_media_commands(
        [{"action": "media", "operation": "play", "query": "hanuman chalisa", "provider": "spotify"}],
        user_directed=True,
    )
    assert [c.get("tool") or c["action"] for c in commands] == ["open", "computer.wait", "media.ui_play"]
    assert commands[-1]["process"] == "Spotify"


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("Paus the song", "pause"),
        ("okay okay paus", "pause"),
        ("pause the song", "pause"),
        ("gaana rok do", "pause"),
        ("resume the song", "play"),
        ("Change the song", "next"),
        ("play the next song", "next"),
        ("agla gaana", "next"),
        ("play the previous song", "previous"),
        ("go back to the previous song", "previous"),
        ("pichla gaana", "previous"),
        ("what song is this", "now_playing"),
        ("who sings this", "now_playing"),
    ],
)
def test_transport_commands_route_without_vision(text, operation):
    entities = IntentRouter().analyze(text).entities
    assert entities == {"action": "media", "operation": operation}


def test_transport_keeps_spotify_and_resume_phrases():
    router = IntentRouter()
    assert router.analyze("pause spotify").entities["provider"] == "spotify"
    assert router.analyze("play the last song I was listening to").entities == {
        "action": "media", "operation": "play"
    }
    assert router.analyze("change the wallpaper").entities.get("action") != "media"


def test_vague_music_query_is_cleaned():
    entities = IntentRouter().analyze("Play something Egyptian music type on Spotify").entities
    assert entities["query"] == "egyptian music"


def test_media_session_messages_and_provider(monkeypatch):
    import core.media.providers as providers
    import core.media.smtc as smtc
    from core.media import MediaRequest

    calls = []

    def fake_session(op, app, **_kw):
        calls.append((op, app))
        return {"ok": True, "app": "Spotify.exe", "title": "Bateu", "artist": "Masoom Sharma", "status": "Playing"}

    monkeypatch.setattr(smtc, "media_session", fake_session)
    monkeypatch.setattr(providers.platform, "system", lambda: "Windows")
    spotify = providers.SpotifyProvider()
    spotify.client_id = ""
    result = spotify.execute(MediaRequest(operation="next"))
    assert result.success and result.message == "Now playing Bateu by Masoom Sharma."
    assert calls[-1] == ("next", "spotify")
    assert smtc.spoken_message("pause", {"title": "Bateu", "artist": "X"}) == "Paused Bateu by X."
    assert smtc.spoken_message("now_playing", {"title": "Bateu", "artist": "X", "status": "Paused"}) == "Bateu by X is paused."
    assert spotify.execute(MediaRequest(operation="now_playing")).message == "This is Bateu by Masoom Sharma."
    assert calls[-1] == ("", "spotify")


def test_media_session_falls_back_to_media_keys(monkeypatch):
    import core.media.providers as providers
    import core.media.smtc as smtc
    from core.media import MediaRequest

    monkeypatch.setattr(smtc, "media_session", lambda *a, **k: {"ok": False, "error": "no_session"})
    monkeypatch.setattr(providers.platform, "system", lambda: "Windows")
    pressed = []
    fake_user32 = SimpleNamespace(keybd_event=lambda vk, *a: pressed.append(vk))
    monkeypatch.setattr(providers.ctypes, "windll", SimpleNamespace(user32=fake_user32), raising=False)
    result = providers.WindowsMediaProvider().execute(MediaRequest(operation="pause"))
    assert result.success and pressed[0] == 0xB3


def test_superseded_failure_is_not_spoken():
    from ai.runtime_patch import _is_superseded_failure
    from core.contracts import PlanStep

    step = PlanStep(id="step-3", description="ui_play", status=PlanStepStatus.COMPLETED)
    plan = SimpleNamespace(get_step=lambda _id: step)
    task = SimpleNamespace(plan=plan)
    ai = SimpleNamespace(kernel=SimpleNamespace(task_manager=SimpleNamespace(get=lambda _id: task)))
    failed = ToolResult(success=False, tool="media.ui_play", metadata={"task_id": "t", "plan_step_id": "step-3"})
    assert _is_superseded_failure(ai, failed)
    step.status = PlanStepStatus.FAILED
    assert not _is_superseded_failure(ai, failed)


@pytest.mark.parametrize(
    ("word", "roman"),
    [("बैठी", "baithi"), ("है", "hai"), ("धड़कन", "dhadkan"), ("कमला", "kamla"),
     ("नमस्ते", "namaste"), ("लड़की", "ladki"), ("तेरी", "teri"), ("प्यार", "pyar")],
)
def test_romanize_hinglish(word, roman):
    from core.transliteration import romanize_hinglish

    assert romanize_hinglish(word) == roman


@pytest.mark.parametrize(
    ("english", "hindi", "expected"),
    [
        ("Can you play bate on Spotify", "कैन यू प्ले बैठी है ऑन स्पॉटिफाई", "Can you play baithi hai on Spotify"),
        # English titles and correctly heard Hindi stay as the English decode.
        ("Can you play blinding lights on Spotify", "कैन यू प्ले ब्लाइंडिंग लाइट्स ऑन स्पॉटिफाई", "Can you play blinding lights on Spotify"),
        ("Play tum hi ho on Spotify", "प्ले तुम ही हो ऑन स्पॉटिफाई", "Play tum hi ho on Spotify"),
    ],
)
def test_hinglish_titles_come_from_the_hindi_decode(english, hindi, expected):
    from voice.language_choice import choose_transcript

    assert choose_transcript(english, hindi) == ("en", expected)


@pytest.mark.parametrize(
    "text",
    ["बैठी है स्पॉटिफाई पर चलाओ", "spotify pe baithi hai chalao"],
)
def test_hindi_play_grammar_names_the_app(text):
    entities = IntentRouter().analyze(text).entities
    assert entities["query"] == "baithi hai" and entities["provider"] == "spotify"


def test_vad_keeps_live_onset_after_post_tts_seed():
    from voice.vad_engine import VADEngine

    vad = VADEngine.__new__(VADEngine)
    vad.pre_roll_samples = 4
    vad.sample_rate = 16000
    vad.start_chunk_rms = 0.0
    vad.min_rms = 0.0
    vad.min_peak = 0.0
    vad.min_speech_duration = 0.0
    import itertools
    import time as _time

    events = itertools.chain([None, None, {"start": 0}], itertools.repeat({"end": 0}))
    vad.debug = False

    class FakeIterator:
        def __call__(self, _tensor):
            return next(events, None)

        def reset_states(self):
            pass

    vad.vad = FakeIterator()

    class Tracker:
        def update(self, *, started, ended, chunk_samples, get_audio):
            return ended

    vad.new_turn_tracker = lambda **_kw: Tracker()
    chunks = itertools.chain(
        [np.full(2, 0.1, np.float32), np.full(2, 0.2, np.float32), np.full(2, 0.3, np.float32)],
        itertools.repeat(np.full(2, 0.4, np.float32)),
    )

    def get_chunk():
        _time.sleep(0.05)
        return next(chunks)

    mic = SimpleNamespace(get_chunk=get_chunk)
    captured = []
    vad.collect_utterance(mic, initial_audio=np.zeros(2, np.float32), on_audio=captured.append)
    audio = np.concatenate(captured)
    # The two chunks heard before Silero's start event (the onset) are kept.
    assert 0.1 in audio and 0.2 in audio


@pytest.mark.parametrize(
    ("english", "hindi", "expected"),
    [
        # "play," with a comma and a garbled "on" ("brought spotify").
        ("Okay play, First Nana brought spotify", "ओके प्ले फिर से नैना भरे उन स्पॉटिफाई",
         "Okay play fir se naina bhare on spotify"),
        # The English decode lost the verb entirely ("Plena").
        ("Plena bre on Spotify", "प्लेन है ना भरे ऑन स्पॉटिफाई", "play hai na bhare on spotify"),
        # Devanagari-spelled English titles keep the English decode.
        ("play shape of you on spotify", "प्ले शेप ऑफ यू ऑन स्पॉटिफाई", "play shape of you on spotify"),
        ("play thunder on spotify", "प्ले थंडर ऑन स्पॉटिफाई", "play thunder on spotify"),
        ("search for human challenges on chrome", "सर्च फॉर हनुमान चलीसा ऑन क्रोम",
         "search for hanuman chalisa on chrome"),
    ],
)
def test_generic_hinglish_queries(english, hindi, expected):
    from voice.language_choice import choose_transcript

    assert choose_transcript(english, hindi) == ("en", expected)


@pytest.mark.parametrize(
    ("word", "hindi"),
    [("भरे", True), ("से", True), ("नैना", True), ("आँखों", True), ("ब्लाइंडिंग", False),
     ("स्पॉटिफाई", False), ("शेप", False)],
)
def test_is_hindi_word(word, hindi):
    from core.transliteration import is_hindi_word

    assert is_hindi_word(word) is hindi


def test_plane_is_not_a_play_command():
    from core.transliteration import devanagari_play_command

    assert devanagari_play_command("प्लेन कब आएगा") is None


def test_language_policy_follows_hindi_voice(monkeypatch):
    from ai.ai_module import _language_policy

    monkeypatch.setenv("ASTA_TTS_HINDI", "1")
    assert "Devanagari" in _language_policy()
    monkeypatch.setenv("ASTA_TTS_HINDI", "0")
    assert "Latin script" in _language_policy()


@pytest.mark.parametrize(
    ("text", "artist"),
    [
        ("play some udit narayan songs on spotify", "udit narayan"),
        ("play sam uditnarayan music on spotify", "udit narayan"),
        ("play songs by arijit singh on spotify", "arijit singh"),
        ("udit narayan ke gaane spotify pe chalao", "udit narayan"),
        ("play romantic songs of kumar sanu on spotify", "kumar sanu"),
    ],
)
def test_artist_requests_play_the_artist(text, artist):
    entities = IntentRouter().analyze(text).entities
    assert entities["query"] == artist and entities["artist"] == artist


@pytest.mark.parametrize(
    "text", ["play egyptian music on spotify", "play sad songs on spotify", "play baithi hai on spotify"]
)
def test_genres_and_titles_are_not_artists(text):
    assert "artist" not in IntentRouter().analyze(text).entities


@pytest.mark.parametrize(
    ("english", "hindi", "expected"),
    [
        ("m on spotify", "उदितनारायण म्यूजिक ऑन स्पॉटिफाई", "play uditnarayan music on spotify"),
        ("Playsm Udnarayan music on Spotify", "प्लेसम उदितनारायण म्यूजिक ऑन स्पॉटिफाई",
         "play sam uditnarayan music on spotify"),
        ("Okay, can you play First Nana bre on Spotify", "ओके, कैन यू प्लेट से नैना भरे ऑन स्पॉटिफाई",
         "Okay, can you play se naina bhare on Spotify"),
    ],
)
def test_hindi_decode_rescues_clipped_play(english, hindi, expected):
    from voice.language_choice import choose_transcript

    assert choose_transcript(english, hindi) == ("en", expected)


def test_ui_play_prefers_the_shortest_full_match_for_artists():
    from core.tools.ui_play import query_tokens

    assert query_tokens("udit narayan") == ["udit", "narayan"]
