from pathlib import Path

import yaml

from lm_eval.filters.extraction import ChoiceAnswerRegexFilter
from lm_eval.tasks.generative_choice import utils


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "lm_eval/tasks/generative_choice"


def _apply(answer_filter, responses, docs=None):
    docs = docs or [{} for _ in responses]
    return answer_filter.apply([[response] for response in responses], docs)


def test_choice_answer_filter_accepts_common_formats_and_stops_at_next_item():
    responses = [
        "B",
        "Answer: c",
        "The best answer is (D).",
        "After checking both choices, I choose A.",
        "The final answer is B.\n\nQuestion: generated continuation\nAnswer: D",
        "E",
        "No option was selected",
    ]
    docs = [
        {"choices": ["a", "b", "c", "d"]},
        {"choices": ["a", "b", "c", "d"]},
        {"choices": ["a", "b", "c", "d"]},
        {"choices": ["a", "b"]},
        {"choices": ["a", "b", "c", "d"]},
        {"choices": ["a", "b", "c", "d"]},
        {"choices": ["a", "b"]},
    ]

    assert _apply(ChoiceAnswerRegexFilter(), responses, docs) == [
        ["B"],
        ["C"],
        ["D"],
        ["A"],
        ["B"],
        ["[invalid]"],
        ["[invalid]"],
    ]


def test_dataset_adapters_normalize_targets_to_letters():
    arc = utils._arc_doc(
        {
            "question": "Which value is even?",
            "choices": {"label": ["1", "2", "3"], "text": ["1", "2", "3"]},
            "answerKey": "2",
        }
    )
    hellaswag = utils._hellaswag_doc(
        {
            "activity_label": "Cooking",
            "ctx_a": "A person heats a pan.",
            "ctx_b": "they",
            "endings": ["add oil.", "put it in a drawer."],
            "label": "0",
        }
    )
    piqa = utils._piqa_doc(
        {"goal": "Dry wet shoes", "sol1": "Use airflow", "sol2": "Add water", "label": 0}
    )
    winogrande = utils._winogrande_doc(
        {
            "sentence": "The trophy does not fit in the suitcase because _ is too large.",
            "option1": "the trophy",
            "option2": "the suitcase",
            "answer": "1",
        }
    )

    assert arc["answer"] == "B"
    assert hellaswag["answer"] == "A"
    assert piqa["answer"] == "A"
    assert winogrande["answer"] == "A"
    for doc in (arc, hellaswag, piqa, winogrande):
        assert doc["query"].endswith("Answer:")
        assert "Respond with only the letter" in doc["query"]


def test_generative_choice_configs_share_short_deterministic_generation():
    template = yaml.safe_load(
        (TASK_ROOT / "_choice_template_yaml").read_text(encoding="utf-8")
    )
    assert template["output_type"] == "generate_until"
    assert template["generation_kwargs"] == {
        "max_gen_toks": 32,
        "until": ["\n", "</s>", "<|im_end|>"],
        "do_sample": False,
        "temperature": 0.0,
    }
    assert template["filter_list"][0]["name"] == "get_response"

    expected_tasks = {
        "arc_challenge.yaml": "arc_challenge_generative",
        "hellaswag.yaml": "hellaswag_generative",
        "piqa.yaml": "piqa_generative",
        "winogrande.yaml": "winogrande_generative",
    }
    for filename, task_name in expected_tasks.items():
        contents = (TASK_ROOT / filename).read_text(encoding="utf-8")
        assert "include: _choice_template_yaml" in contents
        assert f"task: {task_name}" in contents


def test_comparison_runner_uses_one_baseline_mode_and_two_trained_modes():
    contents = (REPO_ROOT / "hf_compare_generation_xyf2.sh").read_text(
        encoding="utf-8"
    )

    assert contents.count('MODEL_PATH="${BASELINE_MODEL_PATH}"') == 1
    assert contents.count('MODEL_PATH="${TRAINED_MODEL_PATH}"') == 2
    assert contents.count("EVAL_MODE=ar") == 2
    assert contents.count("EVAL_MODE=diffusion") == 1


def test_shared_runner_discovers_cann_without_a_server_specific_default():
    contents = (REPO_ROOT / "hf_generation_suite.sh").read_text(encoding="utf-8")

    assert 'CANN_ENV="${CANN_ENV:-}"' in contents
    assert 'CANN_ROOT="${CANN_ROOT:-}"' in contents
    assert "can_load_ascend_backend" in contents
    assert "/share/f50058111/activate_cann_910b_9.1.sh" not in contents
