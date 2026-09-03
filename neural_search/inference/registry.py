"""Environment-driven provider and model registry."""

from __future__ import annotations

import os

from neural_search.inference.schemas import InferenceCapability, ModelProfile, ProviderSettings


class InferenceRegistry:
    """Registry of provider settings and model capability profiles."""

    def __init__(
        self,
        providers: dict[str, ProviderSettings] | None = None,
        models: dict[str, ModelProfile] | None = None,
    ) -> None:
        self.providers = providers or {}
        self.models = models or {}

    @classmethod
    def from_env(cls) -> InferenceRegistry:
        """Build the default registry without requiring any credentials.

        NIM configuration is enabled when ``NIM_BASE_URL`` is present.  ``NVIDIA_API_KEY``
        or ``NIM_API_KEY`` may be used for hosted/authenticated endpoints. Model names are
        intentionally environment-configured because available NIMs differ by deployment.
        """

        providers: dict[str, ProviderSettings] = {}
        models: dict[str, ModelProfile] = {}
        base_url = os.getenv("NIM_BASE_URL")
        if base_url:
            providers["nim"] = ProviderSettings(
                name="nim",
                kind="nim",
                base_url=base_url,
                api_key=os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY"),
                timeout_seconds=float(os.getenv("NIM_TIMEOUT_SECONDS", "120")),
            )
            configured = {
                "scientific_extraction": (
                    os.getenv("NIM_EXTRACTION_MODEL"),
                    {InferenceCapability.CHAT, InferenceCapability.STRUCTURED_EXTRACTION},
                ),
                "code_reasoning": (
                    os.getenv("NIM_CODE_MODEL"),
                    {
                        InferenceCapability.CHAT,
                        InferenceCapability.CODE_REASONING,
                        InferenceCapability.TOOL_CALLING,
                    },
                ),
                "mathematical_review": (
                    os.getenv("NIM_MATH_MODEL"),
                    {InferenceCapability.CHAT, InferenceCapability.MATHEMATICAL_REVIEW},
                ),
            }
            for profile_name, (model, capabilities) in configured.items():
                if model:
                    models[profile_name] = ModelProfile(
                        name=profile_name,
                        provider="nim",
                        model=model,
                        capabilities=capabilities,
                    )
        return cls(providers=providers, models=models)

    def model_for_capability(self, capability: InferenceCapability) -> ModelProfile:
        candidates = [profile for profile in self.models.values() if capability in profile.capabilities]
        if not candidates:
            raise LookupError(f"no model configured for capability {capability.value}")
        return sorted(candidates, key=lambda profile: profile.name)[0]

    def get_model(self, name: str) -> ModelProfile:
        try:
            return self.models[name]
        except KeyError as exc:
            raise LookupError(f"unknown model profile: {name}") from exc

    def get_provider(self, name: str) -> ProviderSettings:
        try:
            return self.providers[name]
        except KeyError as exc:
            raise LookupError(f"unknown inference provider: {name}") from exc

    def describe(self) -> dict[str, object]:
        return {
            "providers": {
                name: {
                    "kind": provider.kind,
                    "base_url": provider.base_url,
                    "authenticated": bool(provider.api_key),
                }
                for name, provider in self.providers.items()
            },
            "models": {
                name: {
                    "provider": profile.provider,
                    "model": profile.model,
                    "capabilities": sorted(capability.value for capability in profile.capabilities),
                }
                for name, profile in self.models.items()
            },
        }
