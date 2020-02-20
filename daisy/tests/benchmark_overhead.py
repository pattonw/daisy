import daisy
from daisy.blocks import create_dependency_graph

import pytest

import logging

logger = logging.getLogger(__name__)
daisy.scheduler._NO_SPAWN_STATUS_THREAD = True


@pytest.mark.benchmark
def test_create_dep_graph(benchmark, caplog):
    caplog.set_level(logging.INFO)

    total_roi = daisy.Roi((0, 0, 0), (42, 42, 42))
    read_roi = daisy.Roi((0, 0, 0), (4, 4, 4))
    write_roi = daisy.Roi((0, 0, 0), (2, 2, 2))

    blocks = benchmark(
        create_dependency_graph,
        **{
            "total_roi": total_roi,
            "block_read_roi": read_roi,
            "block_write_roi": write_roi,
            "read_write_conflict": True,
            "fit": "valid",
        },
    )

    assert len(blocks) == 8000
