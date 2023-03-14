import gc
import time
import random
from typing import Dict, List, Optional, Tuple
from hict.api.ContactMatrixFacet import ContactMatrixFacet
from hict.core.common import QueryLengthUnit, ScaffoldDescriptor
from hict.core.scaffold_tree import ScaffoldTree
import numpy as np
from readerwriterlock import rwlock
from pathlib import Path
import pytest
from pytest import fail
from hypothesis import given, example, event, settings, strategies as st, assume, HealthCheck
from hypothesis.extra import numpy as nps
import multiprocessing
import multiprocessing.managers

mp_manager: multiprocessing.managers.SyncManager = multiprocessing.Manager()

mp_rlock = mp_manager.RLock()


def get_lock():
    return mp_rlock


# random.seed(int(time.time()))

def build_tree(
    scaffold_descriptors: List[ScaffoldDescriptor],
    scaffold_lengths: List[int],
    empty_space_lengths: List[int],
    mp_manager: Optional[multiprocessing.managers.SyncManager]
) -> ScaffoldTree:
    tree = ScaffoldTree(
        assembly_length_bp=np.int64(sum(scaffold_lengths)+sum(empty_space_lengths)),
        mp_manager=mp_manager
    )
    last_pos: np.int64 = np.int64(0)
    for i, sd in enumerate(scaffold_descriptors):
        tree.add_scaffold(
            last_pos+empty_space_lengths[i],
            last_pos+empty_space_lengths[i]+scaffold_lengths[i],
            sd
        )
        last_pos += empty_space_lengths[i]+scaffold_lengths[i]
    return tree


@settings(
    max_examples=5000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    scaffold_descriptors=st.lists(
        st.builds(
            ScaffoldDescriptor.make_scaffold_descriptor,
            scaffold_id=st.integers(0, 10000),
            scaffold_name=st.text(max_size=10),
            spacer_length=st.integers(min_value=500, max_value=501),
        ),
        unique_by=(lambda sd: sd.scaffold_id)
    ),
    scaffold_size_bound=st.integers(min_value=2, max_value=100000),
    empty_size_bound=st.integers(min_value=2, max_value=100000)
)
def test_build_tree(
    scaffold_descriptors: List[ScaffoldDescriptor],
    scaffold_size_bound: int,
    empty_size_bound: int
):
    scaffold_lengths = list(np.random.randint(
        1,
        scaffold_size_bound,
        size=len(scaffold_descriptors),
        dtype=np.int64)
    )
    empty_space_lengths = list(np.random.randint(
        0,
        empty_size_bound,
        size=1+len(scaffold_descriptors),
        dtype=np.int64
    ))

    tree = build_tree(
        scaffold_descriptors=scaffold_descriptors,
        scaffold_lengths=scaffold_lengths,
        empty_space_lengths=empty_space_lengths,
        mp_manager=None
    )
    
    total_assembly_length = sum(scaffold_lengths)+sum(empty_space_lengths)
    
    assert (
        tree.root.subtree_length_bp == total_assembly_length
    ), "Tree does not cover all assembly?"

    nodes: List[ScaffoldTree.Node] = []

    def traverse_fn(node: ScaffoldTree.Node):
        nodes.append(node)

    tree.traverse(traverse_fn)

    expected_descriptors = sorted(
        scaffold_descriptors,
        key=lambda d: d.scaffold_id
    )
    actual_descriptors = sorted(
        map(
            lambda n: n.scaffold_descriptor,
            filter(
                lambda n: n.scaffold_descriptor is not None,
                nodes
            )
        ),
        key=lambda d: d.scaffold_id
    )

    assert (
        expected_descriptors == actual_descriptors
    ), "Not all descriptors are present after building tree??"
