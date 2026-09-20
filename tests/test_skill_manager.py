from core import Kernel, SkillManager
from core.contracts import IntentResult, IntentType, SkillDescriptor
from core.skills import register_builtin_skills


def _debug_intent(text="debug the python error"):
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text=text,
        entities={"action": "debug", "target": "python error"},
        requires_tools=False,
    )


def test_skill_manager_registers_and_discovers_matching_skill():
    manager = SkillManager()
    manager.register(
        SkillDescriptor(
            name="demo_debug",
            description="Debug a demo system.",
            instructions="Inspect, hypothesize, change one thing, verify.",
            triggers=("debug",),
            priority=5,
        )
    )

    skills = manager.discover(_debug_intent())

    assert len(skills) == 1
    assert skills[0].name == "demo_debug"


def test_builtin_skills_cover_software_debugging():
    manager = SkillManager()
    register_builtin_skills(manager)

    skills = manager.discover(_debug_intent())

    names = {skill.name for skill in skills}
    assert "software_debugging" in names


def test_skill_manager_supports_query_discovery_and_providers():
    manager = SkillManager()

    manager.register(
        SkillDescriptor(
            name="demo_search",
            description="Search a demo source.",
            instructions="Search, inspect, summarize.",
            triggers=("search",),
        )
    )

    manager.register_provider(
        "external",
        lambda intent, query: (
            SkillDescriptor(
                name="external_skill",
                description="An external debugging skill.",
                instructions="Use external guidance.",
                provider="external",
            ),
        ),
    )

    query_results = manager.discover(query="demo")
    external_results = manager.discover(query="external")

    assert query_results[0].name == "demo_search"
    assert external_results[0].name == "external_skill"
    assert manager.unregister_provider("external") is True


def test_kernel_owns_skill_manager():
    kernel = Kernel()
    assert isinstance(kernel.skill_manager, SkillManager)
