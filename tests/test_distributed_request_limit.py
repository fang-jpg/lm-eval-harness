from itertools import islice

import pytest

import lm_eval.api.task as task_module
from lm_eval.api.instance import Instance
from lm_eval.api.task import Task, _rank_slice_limit
from lm_eval.config.task import TaskConfig


class _DummyTask(Task):
    def __init__(self):
        self._config = TaskConfig(
            {
                "task": "distributed_limit_test",
                "num_fewshot": 0,
                "output_type": "generate_until",
                "repeats": 1,
            }
        )
        self._instances = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return [{"id": index} for index in range(64)]

    def doc_to_text(self, doc):
        return str(doc["id"])

    def doc_to_target(self, doc):
        return str(doc["id"])

    def fewshot_context(self, doc, **kwargs):
        return self.doc_to_text(doc)

    def construct_requests(self, doc, ctx, **kwargs):
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {}),
            idx=0,
            metadata=kwargs["metadata"],
        )

    def process_results(self, doc, results):
        return {}

    def aggregation(self):
        return {}

    def higher_is_better(self):
        return {}


@pytest.mark.parametrize(
    ("limit", "rank", "world_size", "expected"),
    [
        (None, 0, 8, None),
        (0, 0, 8, None),
        (16, 0, 8, 2),
        (16, 7, 8, 2),
        (17, 0, 8, 3),
        (17, 1, 8, 2),
        (3, 2, 8, 1),
        (3, 3, 8, 0),
    ],
)
def test_rank_slice_limit(limit, rank, world_size, expected):
    assert (
        _rank_slice_limit(limit, rank=rank, world_size=world_size) == expected
    )


@pytest.mark.parametrize("limit", [16, 17])
def test_cached_rank_slices_match_uncached_global_limit(limit):
    documents = list(range(64))
    world_size = 8

    cached_slices = []
    uncached_slices = []
    for rank in range(world_size):
        complete_rank_cache = list(islice(documents, rank, None, world_size))
        rank_limit = _rank_slice_limit(limit, rank=rank, world_size=world_size)
        cached_slices.append(complete_rank_cache[:rank_limit])
        uncached_slices.append(list(islice(documents, rank, limit, world_size)))

    assert cached_slices == uncached_slices
    assert sorted(document for shard in cached_slices for document in shard) == list(
        range(limit)
    )


def test_build_all_requests_preserves_global_limit_on_cache_miss_and_hit(monkeypatch):
    cache_store = {}
    monkeypatch.setattr(
        task_module,
        "load_from_cache",
        lambda file_name, cache: cache_store.get(file_name),
    )
    monkeypatch.setattr(
        task_module,
        "save_to_cache",
        lambda file_name, obj: cache_store.__setitem__(file_name, obj),
    )

    for _ in range(2):
        selected_ids = []
        for rank in range(8):
            task = _DummyTask()
            task.build_all_requests(
                limit=16,
                rank=rank,
                world_size=8,
                cache_requests=True,
            )
            assert len(task.instances) == 2
            selected_ids.extend(instance.doc["id"] for instance in task.instances)
        assert sorted(selected_ids) == list(range(16))
        assert bool(cache_store) is True
