"""Parameter-count regression tests for the nano configs (no data, no training)."""

import pytest

from nanov41.configs import CONFIGS
from nanov41.model import NanoV41


@pytest.mark.parametrize(
    "name,total,non_embedding",
    [
        ("tiny_all_topk", 683_416, 617_880),
        ("d8", 81_620_944, 15_429_584),
        ("d12", 193_408_360, 94_121_320),
    ],
)
def test_param_counts(name: str, total: int, non_embedding: int) -> None:
    model = NanoV41(CONFIGS[name])
    assert model.num_params() == total
    assert model.num_params_non_embedding() == non_embedding


@pytest.mark.parametrize("name", ["tiny", "tiny_all_topk", "d8", "d12"])
def test_configs_build(name: str) -> None:
    model = NanoV41(CONFIGS[name])
    assert model.vocab_size == CONFIGS[name]["vocab_size"]
