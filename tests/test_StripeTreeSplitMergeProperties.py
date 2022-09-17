import math
import random
import time
from typing import List, Dict, Callable, Tuple

import numpy as np
import pytest
from pytest_quickcheck.generator import list_of

from hict.core.common import ContigDescriptor, ContigHideType, ContigDirection, StripeDescriptor
from hict.core.contig_tree import ContigTree
from hict.core.stripe_tree import StripeTree

random.seed(int(time.time()))

max_stripe_size: int = 256


@pytest.mark.randomize(
    log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0, )
@pytest.mark.randomize(
    resolution_index=int, min_num=0, max_num=9)
@pytest.mark.randomize(
    contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(
    contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(
    contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0, max_num=1000)
def test_build_tree(
        log_resolutions,
        resolution_index,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src
):
    resolutions = [np.abs(np.int64(10 ** r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    resolution = resolutions[resolution_index]
    ct, cds, st, sds = build_trees(
        resolutions,
        resolution,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src
    )

    assert st.root is not None, "Stripe tree is empty?"

    assert (
            st.root.get_sizes().block_count == (
        sum([max(1, int(math.ceil(
            cd.contig_length_at_resolution[resolution] / max_stripe_size))) for cd in cds])
    )
    ), "Each contig should be divided into stripes no more that max_stripe_size length"

    assert (
            st.root.get_sizes().length_bins == (
        sum([cd.contig_length_at_resolution[resolution] for cd in cds])
    )
    ), "Stripe tree length in bins must be equal to the total contig length in bins"

    ordered_contig_ids_in_stripe_tree: List[np.int64] = []

    def traverse_fn(node: StripeTree.Node) -> None:
        ctg_id = node.stripe_descriptor.contig_descriptor.contig_id
        if len(ordered_contig_ids_in_stripe_tree) == 0 or ordered_contig_ids_in_stripe_tree[-1] != ctg_id:
            ordered_contig_ids_in_stripe_tree.append(ctg_id)

    st.traverse(traverse_fn)

    assert ordered_contig_ids_in_stripe_tree == [
        cd.contig_id for cd in cds], "Contig order must be preserved"


@pytest.mark.randomize(
    log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0, )
@pytest.mark.randomize(
    resolution_index=int, min_num=0, max_num=9)
@pytest.mark.randomize(
    contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(
    contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(
    contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0, max_num=1000)
@pytest.mark.randomize(left_count=int, min_num=-100, max_num=200)
def test_split_stripe_tree_by_count(
        log_resolutions,
        resolution_index,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src,
        left_count
):
    resolutions = [np.abs(np.int64(10 ** r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    resolution = resolutions[resolution_index]
    ct, cds, st, sds = build_trees(
        resolutions,
        resolution,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src
    )
    stripe_count: int = len(sds)

    assert st.root is not None, "Stripe tree is empty?"
    l, r = st.split_node_by_count(st.root, left_count)

    def generate_traverse_fn(lst: List[StripeDescriptor]) -> Callable[[StripeTree.Node], None]:
        def traverse_fn(node: StripeTree.Node) -> None:
            lst.append(node.stripe_descriptor)

        return traverse_fn

    sdl: List[StripeDescriptor] = []
    sdr: List[StripeDescriptor] = []

    if l is not None:
        StripeTree.traverse_node(l, generate_traverse_fn(sdl))
    if r is not None:
        StripeTree.traverse_node(r, generate_traverse_fn(sdr))

    l_count: np.int64 = (l.get_sizes().block_count if l is not None else 0)
    r_count: np.int64 = (r.get_sizes().block_count if r is not None else 0)

    if left_count < 0:
        assert (
                l_count == 0
        ), "If splitting key is less than zero, there should be empty left subtree after split"
        assert (
                r_count == stripe_count
        ), "If splitting key is less than zero, all nodes should be contained in right subtree after split"
        assert (
                sdl == []
        ), "Left subtree should be empty"
        assert (
                sdr == sds
        ), "All nodes must belong to the right subtree and order should not be changed"
    elif 0 <= left_count < stripe_count:
        assert (
                l_count == left_count
        ), "After split, left subtree should contain requested node count"
        assert (
                r_count == (stripe_count - left_count)
        ), "After split, right subtree should contain all nodes except requested node count"
        assert (
                sdl == sds[:left_count]
        ), "Left subtree should contain requested nodes preserving original order"
        assert (
                sdr == sds[left_count:]
        ), "Right subtree should contain all other nodes preserving original order"
    else:
        assert (
                l_count == stripe_count
        ), "If splitting key is greater or equal than contig count, all nodes should be in left subtree after split"
        assert (
                r_count == 0
        ), "If splitting key is greater or equal than contig count, right subtree must be empty"
        assert (
                sdl == sds
        ), "All contigs must have fallen into left subtree preserving original order"
        assert (
                sdr == []
        ), "Right subtree must be empty"

    st.root = st.merge_nodes(l, r)

    ordered_contig_ids_in_stripe_tree: List[np.int64] = []

    def traverse_fn(node: StripeTree.Node) -> None:
        ctg_id = node.stripe_descriptor.contig_descriptor.contig_id
        if len(ordered_contig_ids_in_stripe_tree) == 0 or ordered_contig_ids_in_stripe_tree[-1] != ctg_id:
            ordered_contig_ids_in_stripe_tree.append(ctg_id)

    st.traverse(traverse_fn)

    assert ordered_contig_ids_in_stripe_tree == [
        cd.contig_id for cd in cds], "Split/merge should not modify original order of contigs"


@pytest.mark.randomize(
    log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0, )
@pytest.mark.randomize(
    resolution_index=int, min_num=0, max_num=9)
@pytest.mark.randomize(
    contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(
    contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(
    contig_index=int, min_num=0, max_num=99)
@pytest.mark.randomize(
    contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0, max_num=1000)
def test_expose_contig_stripes(
        log_resolutions,
        resolution_index,
        contig_directions,
        contig_lengths_bp,
        contig_index,
        contig_lengths_at_resolution_src,
):
    resolutions = [np.abs(np.int64(10 ** r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    resolution = resolutions[resolution_index]
    ct, cds, st, sds = build_trees(
        resolutions,
        resolution,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src
    )
    stripe_count: int = len(sds)

    assert st.root is not None, "Stripe tree is empty?"

    ctgd, location_in_resolutions, location_in_resolutions_excluding_hidden, left_subsize_count = ct.get_contig_location(
        contig_index)

    es = st.expose_segment(
        1 + location_in_resolutions[resolution][0], location_in_resolutions[resolution][1])

    def generate_traverse_fn(lst: List[StripeDescriptor]) -> Callable[[StripeTree.Node], None]:
        def traverse_fn(node: StripeTree.Node) -> None:
            lst.append(node.stripe_descriptor)

        return traverse_fn

    sdl: List[StripeDescriptor] = []
    sds: List[StripeDescriptor] = []
    sdr: List[StripeDescriptor] = []

    if es.less is not None:
        StripeTree.traverse_node(es.less, generate_traverse_fn(sdl))
    if es.segment is not None:
        StripeTree.traverse_node(es.segment, generate_traverse_fn(sds))
    if es.greater is not None:
        StripeTree.traverse_node(es.greater, generate_traverse_fn(sdr))

    l_count: np.int64 = (
        es.less.get_sizes().block_count if es.less is not None else 0)
    s_count: np.int64 = (es.segment.get_sizes(
    ).block_count if es.segment is not None else 0)
    r_count: np.int64 = (es.greater.get_sizes(
    ).block_count if es.greater is not None else 0)

    assert (l_count + s_count +
            r_count) == stripe_count, "ExposedSegment does not contain all stripes?"

    assert all(
        [sd.contig_descriptor.contig_id == contig_index for sd in sds]
    ), "Exposed segment should only contain stripes of that contig"
    assert all(
        [sd.contig_descriptor.contig_id != contig_index for sd in sdl]
    ), "Only exposed segment should contain stripes of that contig"
    assert all(
        [sd.contig_descriptor.contig_id != contig_index for sd in sdr]
    ), "Only exposed segment should contain stripes of that contig"

    st.commit_exposed_segment(es)

    ordered_contig_ids_in_stripe_tree: List[np.int64] = []

    def traverse_fn(node: StripeTree.Node) -> None:
        ctg_id = node.stripe_descriptor.contig_descriptor.contig_id
        if len(ordered_contig_ids_in_stripe_tree) == 0 or ordered_contig_ids_in_stripe_tree[-1] != ctg_id:
            ordered_contig_ids_in_stripe_tree.append(ctg_id)

    st.traverse(traverse_fn)

    assert ordered_contig_ids_in_stripe_tree == [
        cd.contig_id for cd in cds], "Split/merge should not modify original order of contigs"


def build_trees(
        resolutions,
        resolution,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src,
) -> Tuple[ContigTree, List[ContigDescriptor], StripeTree, List[StripeDescriptor]]:
    resolutions = [np.abs(np.int64(r)) for r in resolutions]
    contig_lengths_bp = [np.abs(np.int64(l)) for l in contig_lengths_bp]
    contig_descriptors: List[ContigDescriptor] = []
    contig_lengths_at_resolution: List[Dict[np.int64, np.int64]] = []
    for clr in contig_lengths_at_resolution_src:
        contig_lengths_at_resolution.append(dict())
        for ri, le in enumerate(clr):
            contig_lengths_at_resolution[-1][resolutions[ri]] = np.abs(le)

    contig_count: int = len(contig_lengths_bp)
    for i in range(0, contig_count):
        contig_descriptors.append(ContigDescriptor.make_contig_descriptor(
            i,
            f"ctg-{i}",
            ContigDirection(np.abs(contig_directions[i]) % 2),
            contig_lengths_bp[i],
            contig_lengths_at_resolution[i],
            {res: (ContigHideType.AUTO_HIDDEN if i % 2 == 0 else ContigHideType.FORCED_HIDDEN) if (res != np.int64(
                0) and contig_lengths_bp[i] < res) else (
                ContigHideType.AUTO_SHOWN if i % 2 == 0 else ContigHideType.FORCED_SHOWN) for res in resolutions},
            None
        ))

    ct: ContigTree = ContigTree(np.array(resolutions))
    for i in range(0, contig_count):
        ct.insert_at_position(contig_descriptors[i], i)

    st: StripeTree = StripeTree(resolution)

    stripe_descriptors: List[StripeDescriptor] = []

    stripe_id: int = 0
    for i in range(0, contig_count):
        part_count: int = int(
            math.ceil(contig_lengths_at_resolution[i][resolution] / max_stripe_size))
        for part in range(0, max(1, part_count)):
            sd = StripeDescriptor.make_stripe_descriptor(
                stripe_id,
                max_stripe_size if ((1 + part) * max_stripe_size < contig_lengths_at_resolution[i][resolution]) else (
                        contig_lengths_at_resolution[i][resolution] - (part) * max_stripe_size),
                (max_stripe_size * resolution) if (
                        (1 + part) * max_stripe_size < contig_lengths_at_resolution[i][resolution]) else (
                        contig_lengths_bp[i] % (max_stripe_size * resolution)),
                contig_descriptors[i]
            )
            stripe_descriptors.append(sd)
            st.insert_at_position(stripe_id, sd)
            stripe_id += 1

    return ct, contig_descriptors, st, stripe_descriptors
