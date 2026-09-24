from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@register_filter("regex")
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """Compile `regex_pattern` and set the fallback for non-matches.

        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> Iterable[list[str]]:
        def filter_set(inst: Sequence[str]) -> list[str]:
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        if match:
                            match = match[0]
                        else:
                            match = self.fallback
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        return [filter_set(x) for x in resps]


@register_filter("regex_pos")
class POSFilter(Filter):
    """Extract part-of-speech tags from model responses."""

    def __init__(
        self,
        regex_pattern: str = r"\['(.*?)'\]",
        group_select=0,
        fallback=None,
    ) -> None:
        """Compile `regex_pattern` and set the fallback for non-matches.

        `fallback` defines the output returned if no matches for the regex are located.
        """
        if fallback is None:
            fallback = ["invalid"]
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]):
        def extract_tagged_tokens(text):
            # Extract tagged tokens list from text input using regex
            tokens = re.findall(r"\('([^']*)', '([^']*)'\)", text)
            return [(token, pos) for token, pos in tokens]

        def extract_pos_tags(result):
            pos_tags = []
            if isinstance(result, str):
                result = extract_tagged_tokens(result)
            pos_tags.extend(pos for _, pos in result)
            return pos_tags or self.fallback

        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = extract_pos_tags(resp)
                filtered.append(match)
            return filtered

        return (filter_set(x) for x in resps)


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """Filters out leading and trailing whitespace from responses."""

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        def filter_set(inst: Sequence[str]) -> list[str]:
            return [resp.strip() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("mmlu-answer-regex")
class MMLUAnswerRegexFilter(Filter):
    """Extract an A-D answer using MindSpeed's Qwen3 diffusion rules.

    This mirrors ``extract_answer`` in MindSpeed-LLM's
    ``evaluate_qwen3_diffusion.py``.  Keeping it as an lm-eval filter lets a
    generative MMLU task accept outputs such as ``Answer: B`` and ``(C)``
    without changing the model's generated text.
    """

    _answer_patterns = (
        re.compile(r"(?i)answer\s*is\s*[：:]\s*([A-D])"),
        re.compile(r"(?i)answer\s*[：:]\s*([A-D])"),
        re.compile(r"(?i)^\s*([A-D])\s*[.。]?\s*$"),
        re.compile(r"(?i)[\(（]\s*([A-D])\s*[\)）]"),
        re.compile(r"(?i)([A-D])\s*[.。]?\s*$"),
    )

    def __init__(self, fallback: str = "[invalid]") -> None:
        self.fallback = fallback

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        del docs

        def extract(resp: str) -> str:
            if not isinstance(resp, str):
                return self.fallback
            # A 32-token diffusion block often continues with another MMLU
            # ``Question:`` after emitting the answer.  Only the first
            # non-empty generated line belongs to the current item; scanning
            # the whole block can accidentally capture an A-D label from the
            # generated next question.
            text = resp.strip().splitlines()[0].strip()
            for pattern in self._answer_patterns:
                match = pattern.search(text)
                if match:
                    return match.group(1).upper()
            return self.fallback

        return [[extract(resp) for resp in response_set] for response_set in resps]


@register_filter("boolq-answer-regex")
class BoolQAnswerRegexFilter(Filter):
    """Extract a generated ``yes``/``no`` answer for BoolQ."""

    _answer_patterns = (
        re.compile(r"(?i)answer\s*is\s*[：:]?\s*(yes|no)\b"),
        re.compile(r"(?i)answer\s*[：:]\s*(yes|no)\b"),
        re.compile(r"(?i)^\s*(yes|no)\b"),
        re.compile(r"(?i)\b(yes|no)\s*[.!。]?\s*$"),
    )

    def __init__(self, fallback: str = "[invalid]") -> None:
        self.fallback = fallback

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        del docs

        def extract(resp: str) -> str:
            if not isinstance(resp, str):
                return self.fallback
            # Match the current answer only, not a continuation generated
            # later in the diffusion block.
            text = resp.strip().splitlines()[0].strip()
            for pattern in self._answer_patterns:
                match = pattern.search(text)
                if match:
                    return match.group(1).lower()
            return self.fallback

        return [[extract(resp) for resp in response_set] for response_set in resps]


@register_filter("choice-answer-regex")
class ChoiceAnswerRegexFilter(Filter):
    """Extract a generated multiple-choice answer letter.

    The filter is intentionally shared by the AR and diffusion generative
    tasks.  It accepts direct answers (``B``), labelled answers
    (``Answer: B``), and short explanatory answers ending in a letter.  When
    the processed document exposes a ``choices`` list, letters outside that
    document's actual choice count are rejected.
    """

    _labeled_patterns = (
        re.compile(
            r"(?i)\b(?:the\s+)?(?:final\s+|best\s+)?answer"
            r"(?:\s+is)?\s*[:：]?\s*[\(（\[]?([A-Z])(?![A-Z])[\)）\]]?"
        ),
        re.compile(
            r"(?i)\b(?:the\s+)?(?:best\s+)?(?:choice|option)"
            r"(?:\s+is)?\s*[:：]?\s*[\(（\[]?([A-Z])(?![A-Z])[\)）\]]?"
        ),
        re.compile(
            r"(?i)\b(?:choose|select)(?:d)?\s*[:：]?\s*"
            r"[\(（\[]?([A-Z])(?![A-Z])[\)）\]]?"
        ),
    )
    _direct_pattern = re.compile(r"(?i)^\s*[\(（\[]?([A-Z])[\)）\]]?\s*[.。]?\s*$")
    _terminal_pattern = re.compile(
        r"(?i)(?<![A-Z])[\(（\[]?([A-Z])(?![A-Z])[\)）\]]?\s*[.。]?\s*$"
    )

    def __init__(
        self,
        valid_choices: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        fallback: str = "[invalid]",
    ) -> None:
        normalized = "".join(dict.fromkeys(valid_choices.upper()))
        if not normalized or any(not ("A" <= letter <= "Z") for letter in normalized):
            raise ValueError("valid_choices must contain one or more letters A-Z.")
        self.valid_choices = normalized
        self.fallback = fallback

    @staticmethod
    def _current_answer_segment(response: str) -> str:
        """Drop a hallucinated next item without discarding short reasoning."""
        text = response.strip()
        for separator in (
            "\n\nQuestion:",
            "\nQuestion:",
            "\n\nContext:",
            "\nContext:",
        ):
            text = text.split(separator, 1)[0]
        return text.strip()

    def _valid_for_doc(self, doc: dict[str, Any] | None) -> str:
        if not doc:
            return self.valid_choices
        choices = doc.get("choices")
        if isinstance(choices, (list, tuple)) and choices:
            return self.valid_choices[: len(choices)]
        return self.valid_choices

    def _extract(self, response: str, valid_choices: str) -> str:
        if not isinstance(response, str):
            return self.fallback
        text = self._current_answer_segment(response)
        if not text:
            return self.fallback

        for pattern in self._labeled_patterns:
            matches = [
                match.group(1).upper()
                for match in pattern.finditer(text)
                if match.group(1).upper() in valid_choices
            ]
            if matches:
                return matches[-1]

        direct = self._direct_pattern.fullmatch(text)
        if direct and direct.group(1).upper() in valid_choices:
            return direct.group(1).upper()

        terminal = self._terminal_pattern.search(text)
        if terminal and terminal.group(1).upper() in valid_choices:
            return terminal.group(1).upper()
        return self.fallback

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        documents = list(docs)
        filtered = []
        for index, response_set in enumerate(resps):
            doc = documents[index] if index < len(documents) else None
            valid_choices = self._valid_for_doc(doc)
            filtered.append(
                [self._extract(response, valid_choices) for response in response_set]
            )
        return filtered


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """Extract a model's answer on multiple choice questions with letter answers.

    Assumes each document has a "choices" field containing the list of answer choices
    and that the answer label symbols are of the form (A), (B), (C), ... or A, B, C.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        regexes_to_ignore: list[str] | None = None,
    ) -> None:
        r"""Configure the multi-choice regex filter.

        Args:
            regex_pattern: The basic regex pattern to use. If it fails to match,
                a customized procedure is used:
                step 1 — parse choices between ([A-Z])s and search in the response.
                step 2 — parse with regex ``r'\s*([A-?])'``, where ``?`` varies by
                number of choices.
            group_select: Selects the (group_select)th match from the findall result.
            ignore_case: Ignore case during step 1 matching.
            ignore_punctuation: Remove punctuation during step 1 matching.
            regexes_to_ignore: Remove these regexes during step 1 matching.
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        import unicodedata

        def find_match(regex, resp, convert_dict: dict[str, str] | None = None):
            if convert_dict is None:
                convert_dict = {}
            if not isinstance(resp, str):
                resp = ""
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    non_empty = [m for m in match if m]
                    if not non_empty:
                        return ""
                    match = non_empty[0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs, strict=True):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            # Sort branches longest-first so a choice whose text is a prefix of another
            # choice can't shadow it: regex alternation is leftmost-wins, so without this
            # "Guilty" would match inside "Guilty of Romance". choice_to_alpha is keyed by
            # the choice text, so branch order does not affect the letter assignment.
            fallback_regex = re.compile(
                "|".join(sorted(fallback_regexes, key=len, reverse=True))
            )
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps
