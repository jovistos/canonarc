from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .extractors import OpenAICompatibleExtractor, _strip_json_fences


@dataclass(frozen=True)
class InflectionRequest:
    """Input specification for a single morphological inflection request."""

    name: str
    lang: str
    gender: str | None = None
    kind: str = "character"
    declension: str | None = None


def _coerce_inflection_request(item: InflectionRequest | Mapping[str, Any]) -> InflectionRequest:
    if isinstance(item, InflectionRequest):
        return item
    if isinstance(item, Mapping):
        raw_gender = item.get("gender")
        raw_declension = item.get("declension")
        return InflectionRequest(
            name=str(item["name"]),
            lang=str(item["lang"]),
            gender=str(raw_gender) if raw_gender is not None else None,
            kind=str(item.get("kind") or "character"),
            declension=str(raw_declension) if raw_declension is not None else None,
        )
    raise TypeError(f"Unsupported inflection request item: {type(item).__name__}")


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


def generate_inflections_batch(
    generator: InflectionGenerator,
    items: Sequence[InflectionRequest | Mapping[str, Any]],
    *,
    batch_size: int = 25,
) -> list[list[str]]:
    """Dispatches to `generator.generate_inflections_batch()` when available, with per-item fallback."""
    if not items:
        return []
    requests = [_coerce_inflection_request(item) for item in items]
    batch_fn = getattr(generator, "generate_inflections_batch", None)
    if callable(batch_fn):
        return batch_fn(requests, batch_size=batch_size)
    return [
        generator.generate_inflections(
            req.name,
            req.lang,
            gender=req.gender,
            kind=req.kind,
            declension=req.declension,
        )
        for req in requests
    ]


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


def build_batch_inflection_prompt(items: Sequence[InflectionRequest]) -> str:
    """Builds a compact, indexed JSON batch prompt for multiple morphological inflection requests."""
    payload_items: list[dict[str, Any]] = []
    for idx, req in enumerate(items):
        entry: dict[str, Any] = {
            "id": idx,
            "name": req.name.strip(),
            "lang": req.lang.strip(),
        }
        if req.gender:
            entry["gender"] = req.gender
        if req.declension:
            entry["declension"] = req.declension
        if req.kind and req.kind != "character":
            entry["kind"] = req.kind
        payload_items.append(entry)
    items_json = json.dumps(payload_items, ensure_ascii=False)
    return (
        "For each proper name in the JSON array below, return its common inflected (declined) "
        "and possessive forms used in literary prose (including the base form). "
        'Return ONLY a valid JSON object of the form {"items": [{"id": 0, "forms": [...]}, ...]} '
        "preserving every input id exactly.\n"
        f"Input: {items_json}"
    )


def _parse_batch_inflection_forms(
    raw: str | Sequence[Any] | dict[str, Any],
    chunk: Sequence[InflectionRequest],
) -> dict[int, list[str]]:
    """Parses a batched inflection JSON payload into a mapping of chunk index -> deduplicated forms."""
    parsed: Any = raw
    if isinstance(raw, str):
        cleaned = _strip_json_fences(raw).strip()
        if not cleaned.startswith(("{", "[")):
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            first_bracket = cleaned.find("[")
            last_bracket = cleaned.rfind("]")
            if first_brace != -1 and last_brace > first_brace and (
                first_bracket == -1 or first_brace < first_bracket
            ):
                cleaned = cleaned[first_brace : last_brace + 1]
            elif first_bracket != -1 and last_bracket > first_bracket:
                cleaned = cleaned[first_bracket : last_bracket + 1]
        parsed = json.loads(cleaned)

    results: dict[int, list[str]] = {}

    def _record(idx: int, raw_forms: Any) -> None:
        if 0 <= idx < len(chunk) and idx not in results:
            results[idx] = _parse_inflection_forms(raw_forms, chunk[idx].name)

    if isinstance(parsed, dict):
        container: Any = (
            parsed.get("items")
            or parsed.get("results")
            or parsed.get("entries")
        )
        if container is None:
            # Handle direct {"0": [...], "1": [...]} mapping
            if all(isinstance(k, str) and k.strip().isdigit() for k in parsed):
                container = parsed
            else:
                container = next((v for v in parsed.values() if isinstance(v, (list, dict))), None)
    elif isinstance(parsed, list):
        container = parsed
    else:
        raise ValueError(f"Unsupported batch inflection response payload: {type(parsed).__name__}")

    if isinstance(container, dict):
        for k, v in container.items():
            if isinstance(k, str) and k.strip().isdigit():
                try:
                    _record(int(k.strip()), v)
                except ValueError:
                    continue
    elif isinstance(container, list):
        for pos, entry in enumerate(container):
            if isinstance(entry, dict) and "id" in entry:
                try:
                    idx = int(entry["id"])
                except (TypeError, ValueError):
                    continue
                raw_forms = (
                    entry.get("forms")
                    or entry.get("inflections")
                    or entry.get("variants")
                )
                if raw_forms is not None:
                    try:
                        _record(idx, raw_forms)
                    except ValueError:
                        continue
            elif isinstance(entry, (list, tuple)) and pos < len(chunk):
                try:
                    _record(pos, entry)
                except ValueError:
                    continue
    else:
        raise ValueError("Batch inflection response did not contain a valid items list or id map.")

    return results


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

    def generate_inflections_batch(
        self,
        items: Sequence[InflectionRequest | Mapping[str, Any]],
        *,
        batch_size: int = 25,
    ) -> list[list[str]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        requests = [_coerce_inflection_request(item) for item in items]
        return [
            self.generate_inflections(
                req.name,
                req.lang,
                gender=req.gender,
                kind=req.kind,
                declension=req.declension,
            )
            for req in requests
        ]


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

    def _load_cached(self, key: str, name: str) -> list[str] | None:
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
        return None

    def _save_cached(
        self,
        key: str,
        *,
        name: str,
        lang: str,
        gender: str | None,
        kind: str,
        declension: str | None,
        forms: list[str],
    ) -> None:
        self._memory_cache[key] = list(forms)
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{key}.json"
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
        cached = self._load_cached(key, name)
        if cached is not None:
            return cached

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
        self._save_cached(
            key,
            name=name,
            lang=lang,
            gender=gender,
            kind=kind,
            declension=declension,
            forms=forms,
        )
        return forms

    def generate_inflections_batch(
        self,
        items: Sequence[InflectionRequest | Mapping[str, Any]],
        *,
        batch_size: int = 25,
    ) -> list[list[str]]:
        """Generates inflections in language-grouped batches with shared per-item cache and single-item fallback."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if not items:
            return []

        requests = [_coerce_inflection_request(item) for item in items]
        item_keys: list[str] = []
        resolved_by_key: dict[str, list[str]] = {}
        unique_misses: list[tuple[str, InflectionRequest]] = []
        queued_miss_keys: set[str] = set()

        for req in requests:
            key = self._cache_key(req.name, req.lang, req.gender, req.kind, req.declension)
            item_keys.append(key)
            if key in resolved_by_key:
                continue
            cached = self._load_cached(key, req.name)
            if cached is not None:
                resolved_by_key[key] = cached
                continue
            if key not in queued_miss_keys:
                queued_miss_keys.add(key)
                unique_misses.append((key, req))

        # Group cache misses by normalized language so each prompt stays linguistically coherent
        misses_by_lang: dict[str, list[tuple[str, InflectionRequest]]] = {}
        for key, req in unique_misses:
            misses_by_lang.setdefault(req.lang.strip().lower(), []).append((key, req))

        for lang_group in misses_by_lang.values():
            for start in range(0, len(lang_group), batch_size):
                chunk_pairs = lang_group[start : start + batch_size]
                if len(chunk_pairs) == 1:
                    key, req = chunk_pairs[0]
                    resolved_by_key[key] = self.generate_inflections(
                        req.name,
                        req.lang,
                        gender=req.gender,
                        kind=req.kind,
                        declension=req.declension,
                    )
                    continue

                chunk_reqs = [req for _, req in chunk_pairs]
                prompt = build_batch_inflection_prompt(chunk_reqs)
                batch_max_tokens = (
                    self.max_tokens * len(chunk_reqs)
                    if self.max_tokens is not None
                    else None
                )
                parsed_chunk: dict[int, list[str]] = {}
                for _attempt in range(self.max_retries + 1):
                    content, _ = self.extractor.call_json_api(
                        prompt,
                        system_prompt=self.system_prompt,
                        max_tokens=batch_max_tokens,
                        temperature=self.temperature,
                    )
                    self.api_calls += 1
                    try:
                        parsed_chunk = _parse_batch_inflection_forms(content, chunk_reqs)
                        if parsed_chunk:
                            break
                    except Exception:  # noqa: BLE001
                        parsed_chunk = {}

                # Save successfully parsed items and fall back to per-item calls for any dropped IDs
                for idx, (key, req) in enumerate(chunk_pairs):
                    if idx in parsed_chunk:
                        forms = parsed_chunk[idx]
                        self._save_cached(
                            key,
                            name=req.name,
                            lang=req.lang,
                            gender=req.gender,
                            kind=req.kind,
                            declension=req.declension,
                            forms=forms,
                        )
                        resolved_by_key[key] = forms
                    else:
                        resolved_by_key[key] = self.generate_inflections(
                            req.name,
                            req.lang,
                            gender=req.gender,
                            kind=req.kind,
                            declension=req.declension,
                        )

        return [list(resolved_by_key[key]) for key in item_keys]
