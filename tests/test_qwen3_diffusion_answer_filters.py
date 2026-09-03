from pathlib import Path

import yaml

from lm_eval.filters.extraction import (
    BoolQAnswerRegexFilter,
    MMLUAnswerRegexFilter,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _apply(answer_filter, responses):
    return list(answer_filter.apply([[response] for response in responses], []))


def test_mmlu_answer_filter_matches_mindspeed_extractor():
    responses = [
        " A ",
        "Answer: b",
        "The answer is: C",
        "After checking the options, I choose (D).",
        "Reasoning finished; the answer is A.",
        " D\n\nQuestion: generated continuation\nA. 0\nB. ",
        " B\n\nQuestion: generated continuation\nA. ",
        "No valid option",
    ]

    assert _apply(MMLUAnswerRegexFilter(), responses) == [
        ["A"],
        ["B"],
        ["C"],
        ["D"],
        ["A"],
        ["D"],
        ["B"],
        ["[invalid]"],
    ]


def test_boolq_answer_filter_accepts_direct_and_explained_answers():
    responses = [
        " yes ",
        "No, the passage contradicts it.",
        "Answer: YES",
        "The answer is no.",
        " yes\n\nQuestion: generated continuation",
        "Reasoning only",
    ]

    assert _apply(BoolQAnswerRegexFilter(), responses) == [
        ["yes"],
        ["no"],
        ["yes"],
        ["no"],
        ["yes"],
        ["[invalid]"],
    ]


def test_mmlu_and_boolq_use_one_block_generate_until_configs():
    config_paths = (
        REPO_ROOT / "lm_eval/tasks/mmlu/generative/_default_template_yaml",
        REPO_ROOT / "lm_eval/tasks/super_glue/boolq/seq2seq.yaml",
    )

    for path in config_paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["output_type"] == "generate_until"
        assert config["generation_kwargs"]["max_gen_toks"] == 32
        assert "\n" in config["generation_kwargs"]["until"]
        assert config["generation_kwargs"]["do_sample"] is False
        assert config["generation_kwargs"]["temperature"] == 0.0
        assert config["filter_list"][0]["name"] == "get_response"
