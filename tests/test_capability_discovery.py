from core import Kernel
from core.contracts import IntentResult, IntentType
from core.tools import OpenApplicationTool


def _open_intent():
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.99,
        normalized_text="open calculator",
        entities={"action": "open", "target": "calculator"},
        requires_tools=True,
    )


def test_capability_discovery_ranks_native_capabilities_for_command():
    kernel = Kernel()
    kernel.register_tool(OpenApplicationTool())

    capabilities = kernel.capability_discovery.discover(_open_intent())

    assert capabilities
    assert capabilities[0].name == "system.open_application"
    assert capabilities[0].provider == "native"
    assert capabilities[0].loaded is True


def test_capability_discovery_supports_provider_extensions():
    kernel = Kernel()

    def provider(intent, query):
        return (
            kernel_capability(
                name="demo.search",
                description="Search a demo source.",
                provider="demo",
                loaded=False,
            ),
        )

    kernel.capability_discovery.register_provider("demo", provider)
    capabilities = kernel.capability_discovery.discover(query="demo")

    assert capabilities[0].name == "demo.search"
    assert capabilities[0].provider == "demo"
    assert capabilities[0].loaded is False

    assert kernel.capability_discovery.unregister_provider("demo") is True
    assert "demo" not in kernel.capability_discovery.providers()


def kernel_capability(*, name, description, provider, loaded):
    from core.contracts import CapabilityDescriptor

    return CapabilityDescriptor(
        name=name,
        description=description,
        provider=provider,
        loaded=loaded,
    )
