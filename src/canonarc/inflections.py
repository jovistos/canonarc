from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from .extractors import OpenAICompatibleExtractor, _strip_json_fences


class InflectionGenerator(Protocol):
    """Protocol for one-shot morphological inflection & possessive form generation."""

    def generate_inflections(
        self,
        name: str,
        lang: str,
        *,
        gender: str | None = None,
        kind: str = "character",
        declension: str | None = None,
    ) -> list[str]: ...


def _parse_inflection_forms(raw: str | Sequence[str] | dict[str, Any], base_name: str) -> list[str]:
    """Parses a JSON array, JSON object ({'forms': [...]} / {'inflections': [...]}), or Python sequence."""
    parsed: Any = raw
    if isinstance(raw, str):
        cleaned = _strip_json_fences(raw).strip()
        # Handle bare JSON array embedded in prose
        if not cleaned.startswith(("[", "{")):
            first_bracket = cleaned.find("[")
            last_bracket = cleaned.rfind("]")
            if first_bracket != -1 and last_bracket > first_bracket:
                cleaned = cleaned[first_bracket : last_bracket + 1]
        parsed = json.loads(cleaned)

    raw_items: list[Any]
    if isinstance(parsed, dict):
        extracted = (
            parsed.get("forms")
            or parsed.get("inflections")
            or parsed.get("variants")
            or next((v for v in parsed.values() if isinstance(v, list)), [])
        )
        raw_items = list(extracted) if isinstance(extracted, list) else []
    elif isinstance(parsed, (list, tuple, set)):
        raw_items = list(parsed)
    else:
        raise ValueError(f"Unsupported inflection response payload: {type(parsed).__name__}")

    deduped: list[str] = []
    seen: set[str] = set()

    def _push(val: str) -> None:
        clean = val.strip()
        if not clean:
            return
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(clean)

    _push(base_name)
    for item in raw_items:
        if isinstance(item, str):
            _push(item)
    return deduped


def build_inflection_prompt(
    name: str,
    lang: str,
    *,
    gender: str | None = None,
    kind: str = "character",
    declension: str | None = None,
) -> str:
    """Builds a compact, deterministic one-shot prompt for inflected & possessive forms."""
    hints: list[str] = []
    if gender:
        hints.append(gender)
    if declension:
        hints.append(declension)
    if kind and kind != "character":
        hints.append(kind)
    hint_str = f" ({', '.join(hints)})" if hints else ""
    return (
        f"Given {lang} proper name '{name}'{hint_str}, return a JSON object "
        f'{{"forms": [...]}} containing its common inflected (declined) and possessive forms '
        f"used in literary prose. Include the base form '{name}'. Return ONLY valid JSON."
    )


class CallableInflectionGenerator:
    """Adapter turning any Python callable into an InflectionGenerator with in-memory caching."""

    def __init__(
        self,
        fn: Callable[..., str | Sequence[str] | dict[str, Any]],
        *,
        use_cache: bool = True,
    ) -> None:
        self.fn = fn
        self.use_cache = use_cache
        self._cache: dict[tuple[str, str, str | None, str, str | None], list[str]] = {}
        self._params = list(inspect.signature(fn).parameters.values())

    def _invoke(
        self,
        name: str,
        lang: str,
        gender: str | None,
        kind: str,
        declension: str | None,
    ) -> str | Sequence[str] | dict[str, Any]:
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in self._params)
        param_names = {p.name for p in self._params}
        kwargs: dict[str, Any] = {}
        if accepts_kwargs or "gender" in param_names:
            kwargs["gender"] = gender
        if accepts_kwargs or "kind" in param_names:
            kwargs["kind"] = kind
        if accepts_kwargs or "declension" in param_names:
            kwargs["declension"] = declension
        return self.fn(name, lang, **kwargs)

    def generate_inflections(
        self,
        name: str,
        lang: str,
        *,
        gender: str | None = None,
        kind: str = "character",
        declension: str | None = None,
    ) -> list[str]:
        cache_key = (name.strip(), lang.strip().lower(), gender, kind, declension)
        if self.use_cache and cache_key in self._cache:
            return list(self._cache[cache_key])
        raw = self._invoke(name.strip(), lang.strip(), gender, kind, declension)
        forms = _parse_inflection_forms(raw, name)
        if self.use_cache:
            self._cache[cache_key] = list(forms)
        return forms


class LLMInflectionGenerator:
    """One-shot LLM inflection generator backed by OpenAICompatibleExtractor with memory + disk caching."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a precise morphological inflection generator for literary localization. "
        "Respond ONLY with valid JSON."
    )

    def __init__(
        self,
        extractor: OpenAICompatibleExtractor,
        *,
        cache_dir: str | Path | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 1,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.extractor = extractor
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max(0, max_retries)
        self.system_prompt = system_prompt
        self._memory_cache: dict[str, list[str]] = {}
        self.api_calls: int = 0
        self.cache_hits: int = 0

    def _cache_key(
        self,
        name: str,
        lang: str,
        gender: str | None,
        kind: str,
        declension: str | None,
    ) -> str:
        raw = (
            f"model={self.extractor.model}\0"
            f"name={name.strip()}\0"
            f"lang={lang.strip().lower()}\0"
            f"gender={gender or ''}\0"
            f"kind={kind}\0"
            f"declension={declension or ''}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate_inflections(
        self,
        name: str,
        lang: str,
        *,
        gender: str | None = None,
        kind: str = "character",
        declension: str | None = None,
    ) -> list[str]:
        key = self._cache_key(name, lang, gender, kind, declension)
        if key in self._memory_cache:
            self.cache_hits += 1
            return list(self._memory_cache[key])

        cache_path = (self.cache_dir / f"{key}.json") if self.cache_dir is not None else None
        if cache_path is not None and cache_path.is_file():
            try:
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                forms = _parse_inflection_forms(cached_data, name)
                self._memory_cache[key] = list(forms)
                self.cache_hits += 1
                return forms
            except Exception:  # noqa: BLE001, S110
                pass

        prompt = build_inflection_prompt(
            name.strip(),
            lang.strip(),
            gender=gender,
            kind=kind,
            declension=declension,
        )
        last_exc: Exception | None = None
        forms: list[str] | None = None
        for _attempt in range(self.max_retries + 1):
            content, _ = self.extractor.call_json_api(
                prompt,
                system_prompt=self.system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self.api_calls += 1
            try:
                forms = _parse_inflection_forms(content, name)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if forms is None:
            assert last_exc is not None
            raise last_exc
        self._memory_cache[key] = list(forms)
        if cache_path is not None:
            cache_path.write_text(
                json.dumps(
                    {
                        "name": name.strip(),
                        "lang": lang.strip().lower(),
                        "gender": gender,
                        "kind": kind,
                        "declension": declension,
                        "model": self.extractor.model,
                        "forms": forms,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return forms
