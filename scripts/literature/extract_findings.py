"""CLI: extract scientific findings from OpenAlex paper abstracts.

Usage:
    python scripts/literature/extract_findings.py \\
        --corpus data/corpus/normalized/openalex_neuro \\
        --config configs/literature/finding_extraction_v1.yaml \\
        --out artifacts/literature/findings_v1.jsonl \\
        [--max-papers 10000] \\
        [--resume]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Allow running from repo root without installing the package
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.literature.finding_extractor import (  # noqa: E402
    LLMProviderConfig,
    extract_batch_with_provider,
    extract_findings_from_corpus,
    load_config,
)

_KEY_NAMES = ("CLAUDE_OPUS_API_KEY", "ANTHROPIC_API_KEY")


def _load_env_local(path: Path = _REPO / ".env.local") -> None:
    """Load simple KEY=value entries from .env.local without logging secrets."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _resolve_anthropic_api_key() -> str | None:
    """Return an Anthropic-compatible API key from supported env names."""
    _load_env_local()
    for key_name in _KEY_NAMES:
        value = os.environ.get(key_name)
        if value:
            os.environ["ANTHROPIC_API_KEY"] = value
            return value
    return None


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _openai_base_url(value: str, *, provider: str) -> str:
    value = value.rstrip("/")
    if provider == "openrouter" and value == "https://openrouter.ai":
        return "https://openrouter.ai/api/v1"
    if provider == "xai" and value == "https://api.x.ai":
        return "https://api.x.ai/v1"
    if provider == "poolside" and value == "https://api.poolside.ai":
        return "https://api.poolside.ai/v1"
    return value


def _provider_from_env(
    *,
    name: str,
    key_name: str,
    model: str,
    base_url: str,
    provider: str,
) -> LLMProviderConfig | None:
    key = _env(key_name)
    if not key:
        return None
    return LLMProviderConfig(
        name=name,
        kind="openai_compatible",
        api_key=key,
        model=model,
        base_url=_openai_base_url(base_url, provider=provider),
    )


def discover_ollama_providers(
    base_url: str = "http://localhost:11434",
) -> list[LLMProviderConfig]:
    """Discover locally running Ollama models via /api/tags.

    Returns empty list if Ollama is not running or unreachable.
    Prefers qwen2.5 models first (better instruction-following for JSON extraction).
    """
    try:
        import requests as req
        resp = req.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        models: list[dict] = resp.json().get("models", [])
    except Exception:
        return []

    providers: list[LLMProviderConfig] = []
    for m in models:
        model_name: str = m.get("name", "")
        if not model_name:
            continue
        slug = model_name.replace(":", "-").replace("/", "-")
        providers.append(
            LLMProviderConfig(
                name=f"ollama-{slug}",
                kind="ollama",
                api_key="",
                model=model_name,
                base_url=base_url,
            )
        )

    # Sort so qwen2.5 models come first (structured JSON extraction quality)
    def _rank(p: LLMProviderConfig) -> int:
        n = p.model.lower()
        if "qwen2.5" in n:
            return 0
        if "qwen" in n:
            return 1
        return 2

    return sorted(providers, key=_rank)


def discover_providers() -> list[LLMProviderConfig]:
    """Discover configured LLM providers without logging key material.

    Ollama local models are listed first so they are preferred in auto mode.
    """
    _load_env_local()
    # Local Ollama first (zero cost, deterministic throughput)
    providers: list[LLMProviderConfig] = discover_ollama_providers()

    claude_key = _env("CLAUDE_OPUS_API_KEY") or _env("ANTHROPIC_API_KEY")
    if claude_key:
        providers.append(
            LLMProviderConfig(
                name="anthropic-direct",
                kind="anthropic",
                api_key=claude_key,
                model=_env("ANTHROPIC_MODEL", _env("OPUS_MODEL_NAME", "claude-haiku-4-5-20251001")),
            )
        )

    openrouter_base = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    for name, key_name, model in [
        ("openrouter", "OPENROUTER_API_KEY", _env("OPENROUTER_MODEL", "google/gemini-2.5-flash")),
        (
            "openrouter-owl-alpha",
            "OPENROUTER_OWL_ALPHA_API_KEY1",
            _env("OPENROUTER_OWL_ALPHA_MODEL", "openrouter/horizon-alpha"),
        ),
    ]:
        provider = _provider_from_env(
            name=name,
            key_name=key_name,
            model=model,
            base_url=openrouter_base,
            provider="openrouter",
        )
        if provider:
            providers.append(provider)

    kimi_base = _env("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    for key_name in ["KIMI_API_KEY", "KIMI_API_KEY1", "KIMI_API_KEY2", "KIMI_API_KEY3", "KIMI_API_KEY4"]:
        provider = _provider_from_env(
            name=key_name.lower(),
            key_name=key_name,
            model=_env("KIMI_MODEL", "kimi-k2-0711-preview"),
            base_url=kimi_base,
            provider="kimi",
        )
        if provider:
            providers.append(provider)

    for key_name in ["DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY1"]:
        provider = _provider_from_env(
            name=key_name.lower(),
            key_name=key_name,
            model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            provider="deepseek",
        )
        if provider:
            providers.append(provider)

    for key_name in ["POOLSIDE_LAGUNA_API_KEY", "POOLSIDE_LAGUNA_API_KEY1"]:
        provider = _provider_from_env(
            name=key_name.lower(),
            key_name=key_name,
            model=_env("POOLSIDE_LAGUNA_MODEL", "laguna"),
            base_url=_env("POOLSIDE_BASE_URL", "https://api.poolside.ai/v1"),
            provider="poolside",
        )
        if provider:
            providers.append(provider)

    provider = _provider_from_env(
        name="xai",
        key_name="XAI_API_KEY",
        model=_env("XAI_MODEL", "grok-3-mini"),
        base_url=_env("XAI_BASE_URL", "https://api.x.ai/v1"),
        provider="xai",
    )
    if provider:
        providers.append(provider)

    provider = _provider_from_env(
        name="qwen",
        key_name="QWEN_API_KEY",
        model=_env("QWEN_MODEL", "qwen-plus"),
        base_url=_env("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        provider="qwen",
    )
    if provider:
        providers.append(provider)

    provider = _provider_from_env(
        name="nvidia",
        key_name="NVIDIA_API_KEY",
        model=_env("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
        base_url=_env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        provider="nvidia",
    )
    if provider:
        providers.append(provider)

    return providers


def _probe_paper() -> dict:
    return {
        "paper_id": "probe:paper",
        "paper_doi": None,
        "title": "Hippocampal theta increases during spatial navigation",
        "abstract": (
            "We show that hippocampal theta oscillations increase during active "
            "spatial navigation in mice using electrophysiology."
        ),
    }


def probe_providers(config: dict[str, object], providers: list[LLMProviderConfig]) -> list[LLMProviderConfig]:
    """Try each provider once and return those that produced parseable findings."""
    working: list[LLMProviderConfig] = []
    for provider in providers:
        try:
            findings = extract_batch_with_provider([_probe_paper()], config, provider)
        except Exception:
            findings = []
        if findings:
            logger.info("Provider OK: %s (%s)", provider.name, provider.model)
            working.append(provider)
        else:
            logger.warning("Provider failed/no findings: %s (%s)", provider.name, provider.model)
    return working


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract neuroscience findings from abstracts.")
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Directory containing normalized JSONL shards.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to finding_extraction YAML config.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output JSONL path for extracted findings.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap on number of papers to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skips already-processed papers).",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        help="Provider name to use, or 'auto' to probe and use the first working provider.",
    )
    parser.add_argument(
        "--probe-providers",
        action="store_true",
        help="Probe all discovered providers and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = load_config(args.config)
    if not config:
        logger.error("Config not found or empty: %s", args.config)
        sys.exit(1)

    providers = discover_providers()
    if args.probe_providers:
        if not providers:
            logger.warning("No providers discovered.")
            sys.exit(0)
        working = probe_providers(config, providers)
        logger.info("Working providers: %d/%d", len(working), len(providers))
        sys.exit(0 if working else 1)

    provider: LLMProviderConfig | None = None
    api_key: str | None = None
    if args.provider == "auto":
        working = probe_providers(config, providers)
        if working:
            provider = working[0]
            logger.info("Using provider: %s (%s)", provider.name, provider.model)
        else:
            logger.warning("No working provider found.")
            sys.exit(0)
    elif args.provider == "anthropic":
        api_key = _resolve_anthropic_api_key()
        if not api_key:
            logger.warning(
                "No Anthropic key found — set ANTHROPIC_API_KEY or CLAUDE_OPUS_API_KEY."
            )
            sys.exit(0)
    else:
        provider = next((p for p in providers if p.name == args.provider), None)
        if provider is None:
            logger.error("Provider not found: %s", args.provider)
            logger.error("Discovered providers: %s", ", ".join(p.name for p in providers))
            sys.exit(1)

    corpus_dir: Path = args.corpus
    if not corpus_dir.exists():
        logger.warning("Corpus directory does not exist: %s — nothing to process.", corpus_dir)
        sys.exit(0)

    shards = sorted(corpus_dir.glob("*.jsonl"))
    if not shards:
        logger.warning("No JSONL shards found in %s — nothing to process.", corpus_dir)
        sys.exit(0)

    checkpoint_path: Path | None = None
    if args.resume:
        checkpoint_path = args.out.with_suffix(".checkpoint.json")

    logger.info(
        "Processing %d shards | max_papers=%s | resume=%s",
        len(shards),
        args.max_papers,
        args.resume,
    )

    total = extract_findings_from_corpus(
        shards,
        config,
        args.out,
        checkpoint_path=checkpoint_path,
        max_papers=args.max_papers,
        api_key=api_key,
        provider=provider,
    )

    logger.info("Done. Extracted %d findings -> %s", total, args.out)


if __name__ == "__main__":
    main()
