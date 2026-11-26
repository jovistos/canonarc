from __future__ import annotations

import re
import unicodedata

from ..embedders import _CJK_CHAR_RE
from ..models import ENTITY_KIND_PREFIXES
from ..text_masking import mask_ignored_spans

_ENTITY_ID_PREFIXES: tuple[str, ...] = (
    *dict.fromkeys(f"{prefix}_" for prefix in ENTITY_KIND_PREFIXES.values()),
    "ent_",
)

_LEADING_ARTICLES_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

_SCRIPT_ALIASES: dict[str, str] = {
    "latin": "latin",
    "latn": "latin",
    "cyrillic": "cyrillic",
    "cyrl": "cyrillic",
    "greek": "greek",
    "grek": "greek",
    "arabic": "arabic",
    "arab": "arabic",
    "hebrew": "hebrew",
    "hebr": "hebrew",
    "devanagari": "devanagari",
    "deva": "devanagari",
    "han": "han",
    "cjk": "han",
    "hani": "han",
    "hans": "han",
    "hant": "han",
    "hiragana": "hiragana",
    "hira": "hiragana",
    "katakana": "katakana",
    "kana": "katakana",
    "hangul": "hangul",
    "hang": "hangul",
    "thai": "thai",
    "armenian": "armenian",
    "armn": "armenian",
    "georgian": "georgian",
    "geor": "georgian",
    "bengali": "bengali",
    "beng": "bengali",
    "tamil": "tamil",
    "taml": "tamil",
}

_SUPPORTED_SCRIPTS: frozenset[str] = frozenset(_SCRIPT_ALIASES.values())

_UNAMBIGUOUS_LANG_SCRIPTS: dict[str, tuple[str, ...]] = {
    "ru": ("cyrillic",),
    "uk": ("cyrillic",),
    "bg": ("cyrillic",),
    "be": ("cyrillic",),
    "mk": ("cyrillic",),
    "el": ("greek",),
    "ar": ("arabic",),
    "fa": ("arabic",),
    "ur": ("arabic",),
    "he": ("hebrew",),
    "iw": ("hebrew",),
    "hi": ("devanagari",),
    "th": ("thai",),
    "ko": ("hangul",),
    "ja": ("han", "hiragana", "katakana"),
    "hy": ("armenian",),
    "ka": ("georgian",),
    "bn": ("bengali",),
    "ta": ("tamil",),
}

_URL_OR_EMAIL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b)",
    re.IGNORECASE,
)
_MARKUP_OR_PLACEHOLDER_RE = re.compile(
    r"</?[A-Za-z][^>\n]*>|\{[^{}\s]+\}|%(?:\([^)\s]+\))?[#0\- +]?(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?[diouxXeEfFgGcrsa%]"
)
_ROMAN_NUMERAL_RE = re.compile(
    r"^M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$"
)
_LETTER_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _phrase_pattern(phrase: str) -> str:
    prefix = "" if _CJK_CHAR_RE.match(phrase[0]) else r"(?<!\w)"
    suffix = "" if _CJK_CHAR_RE.match(phrase[-1]) else r"(?!\w)"
    return rf"{prefix}{re.escape(phrase)}{suffix}"


def _extract_snippet(text: str, start: int, end: int, radius: int = 30) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{text[lo:hi]}{suffix}"


def _normalize_label(text: str) -> str:
    cleaned = re.sub(r"[\s_\-]+", " ", text.strip().lower()).strip()
    return _LEADING_ARTICLES_RE.sub("", cleaned).strip()


def _entity_id_slug(entity_id: str) -> str:
    low = entity_id.strip().lower()
    for prefix in _ENTITY_ID_PREFIXES:
        if low.startswith(prefix):
            low = low[len(prefix) :]
            break
    return _normalize_label(low)


def _normalize_script_name(script: str) -> str:
    key = script.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _SCRIPT_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported script '{script}'. Supported scripts: {sorted(_SUPPORTED_SCRIPTS)}."
        )
    return normalized


def _infer_scripts_from_lang(lang: str | None) -> frozenset[str] | None:
    """Infers expected script(s) only from unambiguous BCP-47 tags (e.g. 'sr-Latn', 'ru', 'el')."""
    if not lang or not lang.strip():
        return None
    parts = [p.strip().lower() for p in re.split(r"[-_]", lang.strip()) if p.strip()]
    if not parts:
        return None
    for subtag in parts[1:]:
        if subtag in _SCRIPT_ALIASES:
            return frozenset({_SCRIPT_ALIASES[subtag]})
    primary = parts[0]
    inferred = _UNAMBIGUOUS_LANG_SCRIPTS.get(primary)
    if inferred is not None:
        return frozenset(inferred)
    return None


def _char_script(ch: str) -> str | None:
    """Classifies a Unicode letter into a canonical script name, or None for non-letters/common chars."""
    if not unicodedata.category(ch).startswith("L"):
        return None
    cp = ord(ch)
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
        return "latin"
    if (0x0400 <= cp <= 0x052F) or (0x2DE0 <= cp <= 0x2DFF) or (0xA640 <= cp <= 0xA69F) or (0x1C80 <= cp <= 0x1C8F):
        return "cyrillic"
    if (0x0370 <= cp <= 0x03FF) or (0x1F00 <= cp <= 0x1FFF):
        return "greek"
    if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0x08A0 <= cp <= 0x08FF):
        return "arabic"
    if 0x0590 <= cp <= 0x05FF:
        return "hebrew"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if (0x30A0 <= cp <= 0x30FF) or (0x31F0 <= cp <= 0x31FF) or (0xFF65 <= cp <= 0xFF9F):
        return "katakana"
    if (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F):
        return "hangul"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if _CJK_CHAR_RE.match(ch):
        return "han"

    name = unicodedata.name(ch, "")
    if not name:
        return None
    first_word = name.split(" ", 1)[0].lower()
    if first_word in _SCRIPT_ALIASES:
        return _SCRIPT_ALIASES[first_word]
    if "CJK" in name:
        return "han"
    return None


def _mask_spans(text: str, pattern: re.Pattern[str]) -> str:
    """Replaces matched spans with spaces of identical length (preserving newlines) so character offsets stay aligned."""
    return mask_ignored_spans(text, (pattern,))
