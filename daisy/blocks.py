from .daisy import blocks
from .freezable import Freezable

from .pyblocks import expand_roi_to_grid  # noqa
from .pyblocks import expand_write_roi_to_grid  # noqa
from .pyblocks import get_subgraph_blocks  # noqa
from .pyblocks import expand_request_roi_to_grid  # noqa



def create_dependency_graph(
    total_roi, block_read_roi, block_write_roi, read_write_conflict=True, fit="valid"
):
    dep_graph = blocks.create_dependency_graph(
        total_roi.get_offset(),
        total_roi.get_shape(),
        block_read_roi.get_offset(),
        block_read_roi.get_shape(),
        block_write_roi.get_offset(),
        block_write_roi.get_shape(),
        read_write_conflict,
        fit,
    )
    return dep_graph
