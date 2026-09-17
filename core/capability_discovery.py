from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import CapabilityDescriptor, IntentResult, IntentType
from .tools.selector import ToolSelector


CapabilityProvider = Callable[
    [IntentResult | None, str | None],
    tuple[CapabilityDescriptor, ...],
]


class CapabilityDiscovery:
    """Discover capabilities without coupling reasoning to execution."""

    def __init__(self, registry, *, event_bus=None):
        self.registry = registry
        self.event_bus = event_bus
        self.selector = ToolSelector(registry)
        self._providers: dict[str, CapabilityProvider] = {}

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def register_provider(self, name: str, provider: CapabilityProvider) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError("capability provider name must be non-empty")
        if not callable(provider):
            raise TypeError("capability provider must be callable")
        if key in self._providers:
            raise ValueError(f"Capability provider already registered: {key}")

        self._providers[key] = provider
        self._emit("capability_provider_registered", provider=key)

    def unregister_provider(self, name: str) -> bool:
        key = str(name).strip()
        removed = self._providers.pop(key, None) is not None
        if removed:
            self._emit("capability_provider_unregistered", provider=key)
        return removed

    def providers(self) -> tuple[str, ...]:
        return tuple(self._providers.keys())

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(
        self,
        intent: IntentResult | None = None,
        *,
        query: str | None = None,
        limit: int = 5,
    ) -> tuple[CapabilityDescriptor, ...]:
        limit = max(1, int(limit))
        query_value = str(query).strip().lower() if query else ""

        descriptors: list[CapabilityDescriptor] = []

        if intent is not None and intent.intent is IntentType.COMMAND:
            try:
                ranked = self.selector.rank(intent)
            except ValueError:
                ranked = ()

            for _, _, definition in ranked:
                descriptors.append(
                    CapabilityDescriptor.from_tool_definition(definition)
                )
        else:
            descriptors.extend(
                CapabilityDescriptor.from_tool_definition(definition)
                for definition in self.registry.definitions()
            )

        for provider in self._providers.values():
            try:
                descriptors.extend(provider(intent, query))
            except Exception as exc:
                self._emit(
                    "capability_provider_error",
                    error=f"{type(exc).__name__}: {exc}",
                )

        filtered = self._filter_and_dedupe(descriptors, query_value)
        return tuple(filtered[:limit])

    def discover_commands(
        self,
        intents: list[IntentResult],
        *,
        limit_per_intent: int = 3,
    ) -> tuple[CapabilityDescriptor, ...]:
        results: list[CapabilityDescriptor] = []
        seen: set[str] = set()

        for intent in intents:
            for descriptor in self.discover(intent, limit=limit_per_intent):
                if descriptor.name in seen:
                    continue
                seen.add(descriptor.name)
                results.append(descriptor)

        return tuple(results)

    @staticmethod
    def _filter_and_dedupe(
        descriptors: list[CapabilityDescriptor],
        query: str,
    ) -> list[CapabilityDescriptor]:
        results: list[CapabilityDescriptor] = []
        seen: set[str] = set()

        for descriptor in descriptors:
            if descriptor.name in seen:
                continue

            if query:
                haystack = " ".join(
                    [
                        descriptor.name,
                        descriptor.description,
                        descriptor.provider,
                        " ".join(
                            str(item)
                            for item in (descriptor.metadata.get("actions") or ())
                        ),
                        " ".join(
                            str(item)
                            for item in (descriptor.metadata.get("action_aliases") or ())
                        ),
                    ]
                ).lower()
                if query not in haystack:
                    continue

            seen.add(descriptor.name)
            results.append(descriptor)

        return results

    def _emit(self, event: str, **payload: Any) -> None:
        if self.event_bus is not None:
            self.event_bus.emit(event, **payload)
