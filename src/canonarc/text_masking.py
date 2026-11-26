from __future__ import annotations

import re
from collections.abc import Callable, Sequence

TextNormalizer = Callable[[str], str]

# Vetted, non-greedy built-in markup and placeholder patterns.
# Ordered broadest-first so compound constructs (comments, Markdown images/links)
# are masked before narrower tags or placeholders.
MARKUP_HTML_COMMENTS: re.Pattern[str] = re.compile(r"<!--[\s\S]*?-->")
MARKUP_MARKDOWN_IMAGES: re.Pattern[str] = re.compile(r"!\[[^\]]*\]\([^)\n]*\)")
# Masks only the "(url)" portion of Markdown links "[anchor](url)" so visible anchor prose is preserved.
MARKUP_MARKDOWN_LINK_URLS: re.Pattern[str] = re.compile(r"(?<=\])\([^)\n]*\)")
# Masks opening/closing HTML/XML tags without erasing inner prose (e.g. "<b>Arthur</b>" -> "   Arthur    ").
MARKUP_HTML_TAGS: re.Pattern[str] = re.compile(r"</?[A-Za-z][^>\n]*>")
MARKUP_BRACKET_PLACEHOLDERS: re.Pattern[str] = re.compile(r"\[\[[^\]\n]+\]\]")
MARKUP_BRACE_PLACEHOLDERS: re.Pattern[str] = re.compile(r"\{[^{}\s]+\}")
MARKUP_PRINTF_PLACEHOLDERS: re.Pattern[str] = re.compile(
    r"%(?:\([^)\s]+\))?[#0\- +]?(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?[diouxXeEfFgGcrsa%]"
)

COMMON_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    MARKUP_BRACKET_PLACEHOLDERS,
    MARKUP_BRACE_PLACEHOLDERS,
    MARKUP_PRINTF_PLACEHOLDERS,
)

COMMON_MARKUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    MARKUP_HTML_COMMENTS,
    MARKUP_MARKDOWN_IMAGES,
    MARKUP_MARKDOWN_LINK_URLS,
    MARKUP_HTML_TAGS,
    MARKUP_BRACKET_PLACEHOLDERS,
    MARKUP_BRACE_PLACEHOLDERS,
    MARKUP_PRINTF_PLACEHOLDERS,
)

_HORIZONTAL_SPACES_RE: re.Pattern[str] = re.compile(r"[ \t]{2,}")


def compile_ignore_patterns(
    patterns: Sequence[str | re.Pattern[str]] | None,
) -> tuple[re.Pattern[str], ...]:
    """Compiles a sequence of regex strings or compiled patterns into an immutable tuple."""
    if not patterns:
        return ()
    compiled: list[re.Pattern[str]] = []
    for pat in patterns:
        if isinstance(pat, re.Pattern):
            compiled.append(pat)
        elif isinstance(pat, str):
            if pat:
                compiled.append(re.compile(pat))
        else:
            raise TypeError(
                f"ignore_patterns entries must be str or re.Pattern, got {type(pat).__name__}"
            )
    return tuple(compiled)


def _space_mask_match(match: re.Match[str]) -> str:
    matched = match.group(0)
    if "\n" not in matched:
        return " " * len(matched)
    return "".join("\n" if ch == "\n" else " " for ch in matched)


def mask_ignored_spans(
    text: str,
    patterns: Sequence[re.Pattern[str]],
) -> str:
    """Replaces matched spans with spaces of identical length, preserving newlines.

    Preserves 1:1 character offsets and line breaks so audit spans, snippets,
    and line-anchored patterns remain aligned with the original text.
    """
    if not text or not patterns:
        return text
    masked = text
    for pat in patterns:
        masked = pat.sub(_space_mask_match, masked)
    return masked


def strip_ignored_spans(
    text: str,
    patterns: Sequence[re.Pattern[str]],
    *,
    normalizer: TextNormalizer | None = None,
) -> str:
    """Removes ignored spans and collapses redundant horizontal whitespace while preserving newlines."""
    if not text:
        return text
    cleaned = text
    if patterns:
        cleaned = mask_ignored_spans(cleaned, patterns)
        cleaned = _HORIZONTAL_SPACES_RE.sub(" ", cleaned)
        if "\n" in cleaned:
            cleaned = "\n".join(line.strip(" \t") for line in cleaned.split("\n"))
        else:
            cleaned = cleaned.strip(" \t")
    if normalizer is not None:
        cleaned = normalizer(cleaned)
    return cleaned


def apply_text_masking(
    text: str,
    patterns: Sequence[re.Pattern[str]],
    *,
    normalizer: TextNormalizer | None = None,
) -> str:
    """Applies offset-preserving pattern masking followed by an optional custom normalizer."""
    if not text:
        return text
    out = mask_ignored_spans(text, patterns) if patterns else text
    if normalizer is not None:
        out = normalizer(out)
    return out
