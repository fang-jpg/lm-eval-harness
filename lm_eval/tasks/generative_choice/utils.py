"""Document adapters for short-answer generative choice benchmarks."""

from __future__ import annotations

import re
import string
from typing import Any


CHOICE_LABELS = string.ascii_uppercase


def _format_prompt(instruction: str, question: str, choices: list[str]) -> str:
    if not choices or len(choices) > len(CHOICE_LABELS):
        raise ValueError(f"Unsupported number of choices: {len(choices)}")
    option_lines = "\n".join(
        f"{CHOICE_LABELS[index]}. {choice.strip()}"
        for index, choice in enumerate(choices)
    )
    return (
        f"{instruction}\n"
        f"{question.strip()}\n"
        f"{option_lines}\n"
        "Respond with only the letter of the correct option.\n"
        "Answer:"
    )


def _normalized_doc(instruction: str, question: str, choices: list[Any], answer: int):
    normalized_choices = [str(choice) for choice in choices]
    if not 0 <= answer < len(normalized_choices):
        raise ValueError(
            f"Answer index {answer} is outside {len(normalized_choices)} choices."
        )
    return {
        "query": _format_prompt(instruction, question, normalized_choices),
        "choices": normalized_choices,
        "answer": CHOICE_LABELS[answer],
    }


def _arc_doc(doc: dict[str, Any]) -> dict[str, Any]:
    choices = doc["choices"]
    labels = [str(label).strip() for label in choices["label"]]
    answer_key = str(doc["answerKey"]).strip()
    try:
        answer = labels.index(answer_key)
    except ValueError:
        if answer_key.isdigit() and 1 <= int(answer_key) <= len(labels):
            answer = int(answer_key) - 1
        else:
            raise ValueError(
                f"ARC answer key {answer_key!r} is not present in labels {labels!r}."
            ) from None
    return _normalized_doc(
        "Choose the best answer to the science question.",
        f"Question: {doc['question']}",
        choices["text"],
        answer,
    )


def _hellaswag_text(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _hellaswag_doc(doc: dict[str, Any]) -> dict[str, Any]:
    context = _hellaswag_text(
        f"{doc['activity_label']}: {doc['ctx_a']} {doc['ctx_b'].capitalize()}"
    )
    choices = [_hellaswag_text(ending) for ending in doc["endings"]]
    return _normalized_doc(
        "Choose the most plausible continuation of the context.",
        f"Context: {context}",
        choices,
        int(doc["label"]),
    )


def _piqa_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return _normalized_doc(
        "Choose the most appropriate solution to the physical reasoning problem.",
        f"Question: {doc['goal']}",
        [doc["sol1"], doc["sol2"]],
        int(doc["label"]),
    )


def _winogrande_doc(doc: dict[str, Any]) -> dict[str, Any]:
    answer = int(doc["answer"]) - 1
    return _normalized_doc(
        "Choose the option that best fills the blank in the sentence.",
        f"Sentence: {doc['sentence']}",
        [doc["option1"], doc["option2"]],
        answer,
    )


def process_arc(dataset):
    return dataset.map(_arc_doc)


def process_hellaswag(dataset):
    return dataset.map(_hellaswag_doc)


def process_piqa(dataset):
    return dataset.map(_piqa_doc)


def process_winogrande(dataset):
    return dataset.map(_winogrande_doc)
