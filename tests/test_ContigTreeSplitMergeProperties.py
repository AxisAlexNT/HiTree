import time
import random

random.seed(int(time.time()))

import copy
from typing import List, Dict, Callable, Tuple

import numpy as np
import pytest
from pytest_quickcheck.generator import list_of

from hict.core.common import ContigDescriptor, ContigHideType, ContigDirection
from hict.core.contig_tree import ContigTree


@pytest.mark.randomize(
    log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0,)
@pytest.mark.randomize(
    contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(
    contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(
    contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0, max_num=1000)
def test_build_tree(
        log_resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src
):
    resolutions = [np.abs(np.int64(10**r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    ct, cds = build_tree(
        resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src
    )

    assert ct.get_sizes()[1] == contig_count, "Contig count in tree must be the same as supplied"

    ordered_contig_ids_in_tree: List[np.int64] = []

    def traverse_fn(node: ContigTree.Node) -> None:
        ordered_contig_ids_in_tree.append(node.contig_descriptor.contig_id)

    ct.traverse(traverse_fn)

    assert ordered_contig_ids_in_tree == [cd.contig_id for cd in cds], "Contig order must be preserved"


@pytest.mark.randomize(
    log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0,)
@pytest.mark.randomize(
    contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(
    contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(
    contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0, max_num=1000)
@pytest.mark.randomize(left_length=int, min_num=-100, max_num=200)
def test_split_merge_by_count(
        log_resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src,
        left_length
):
    resolutions = [np.abs(np.int64(10**r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    ct, cds = build_tree(
        resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src
    )
    l, r = ct.split_node_by_count(ct.root, left_length)

    def generate_traverse_fn(lst: List[ContigDescriptor]) -> Callable[[ContigTree.Node], None]:
        def traverse_fn(node: ContigTree.Node) -> None:
            lst.append(node.contig_descriptor)

        return traverse_fn

    cdl: List[ContigDescriptor] = []
    cdr: List[ContigDescriptor] = []

    if l is not None:
        ContigTree.traverse_node(l, generate_traverse_fn(cdl))
    if r is not None:
        ContigTree.traverse_node(r, generate_traverse_fn(cdr))

    l_count: np.int64 = (l.get_sizes()[1] if l is not None else 0)
    r_count: np.int64 = (r.get_sizes()[1] if r is not None else 0)

    if left_length < 0:
        assert (
                l_count == 0
        ), "If splitting key is less than zero, there should be empty left subtree after split"
        assert (
                r_count == contig_count
        ), "If splitting key is less than zero, all nodes should be contained in right subtree after split"
        assert (
                cdl == []
        ), "Left subtree should be empty"
        assert (
                cdr == cds
        ), "All nodes must belong to the right subtree and order should not be changed"
    elif 0 <= left_length < contig_count:
        assert (
                l_count == left_length
        ), "After split, left subtree should contain requested node count"
        assert (
                r_count == (contig_count - left_length)
        ), "After split, right subtree should contain all nodes except requested node count"
        assert (
                cdl == cds[:left_length]
        ), "Left subtree should contain requested nodes preserving original order"
        assert (
                cdr == cds[left_length:]
        ), "Right subtree should contain all other nodes preserving original order"
    else:
        assert (
                l_count == contig_count
        ), "If splitting key is greater or equal than contig count, all nodes should be in left subtree after split"
        assert (
                r_count == 0
        ), "If splitting key is greater or equal than contig count, right subtree must be empty"
        assert (
                cdl == cds
        ), "All contigs must have fallen into left subtree preserving original order"
        assert (
                cdr == []
        ), "Right subtree must be empty"

    ct.root = ct.merge_nodes(l, r)

    new_ord: List[ContigDescriptor] = []

    ct.traverse(generate_traverse_fn(new_ord))

    assert (
            new_ord == cds
    ), "Split/merge should not modify original order of contigs"


@pytest.mark.randomize(
    log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0,)
@pytest.mark.randomize(
    contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(
    contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(
    contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0, max_num=1000)
@pytest.mark.randomize(segment_start=int, min_num=-100, max_num=200)
@pytest.mark.randomize(segment_end=int, min_num=-100, max_num=200)
def test_expose_by_count(
        log_resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src,
        segment_start,
        segment_end,
):
    resolutions = [np.abs(np.int64(10**r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    ct, cds = build_tree(
        resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src,
    )

    if segment_start > segment_end:
        segment_start, segment_end = segment_end, segment_start

    es: ContigTree.ExposedSegment = ct.expose_segment_by_count(segment_start, segment_end)

    def generate_traverse_fn(lst: List[ContigDescriptor]) -> Callable[[ContigTree.Node], None]:
        def traverse_fn(node: ContigTree.Node) -> None:
            lst.append(node.contig_descriptor)

        return traverse_fn

    ord_start = max(0, min(segment_start, contig_count - 1))
    ord_end = max(0, min(segment_end, contig_count - 1))

    expected_segment_length: np.int64 = (ord_end - ord_start + 1)
    if segment_end < 0 or segment_start >= contig_count:
        expected_segment_length = 0

    assert (
            (
                es.segment.get_sizes()[1] if es.segment is not None else 0
            ) == expected_segment_length
    ), "Segment length must be end-start+1 if segment falls into the tree"
    assert (
            (
                es.less.get_sizes()[1] if es.less is not None else 0
            ) == max(0, min(segment_start, contig_count))
    ), "Less contigs should be in exposed segment less"
    if segment_start >= contig_count:
        assert es.greater is None
        assert es.segment is None
    elif segment_end < 0:
        assert es.less is None
        assert es.segment is None
    elif segment_end >= contig_count - 1:
        assert es.greater is None
    else:
        assert (
                es.greater.get_sizes()[1] == (contig_count - ord_end - 1)
        ), "Greater contigs should be in exposed segment greater"

    less_cds: List[ContigDescriptor] = []
    segm_cds: List[ContigDescriptor] = []
    last_cds: List[ContigDescriptor] = []

    ContigTree.traverse_node(es.less, generate_traverse_fn(less_cds))
    ContigTree.traverse_node(es.segment, generate_traverse_fn(segm_cds))
    ContigTree.traverse_node(es.greater, generate_traverse_fn(last_cds))

    assert (
            less_cds == (cds[:ord_start] if segment_start < contig_count else cds)
    ), "Exposed segment's less must contain elements on positions 0...segment_start-1"
    if segment_end < 0 or segment_start >= contig_count:
        assert (
                segm_cds == []
        ), "Segments that are outside of the range, should be empty"
    else:
        assert (
                segm_cds == cds[ord_start:(ord_end + 1)]
        ), "Exposed segment must contain elements on positions segment_start ... segment_end"
    if segment_end < 0:
        assert (
                last_cds == cds
        ), "If segment should end prior to the zero position, all elements are in the greater part"
    elif segment_start >= contig_count:
        assert (
                last_cds == []
        ), "If segment starts after the last possible position, greater side is empty"
    else:
        assert (
                last_cds == cds[(1 + ord_end):]
        ), "Greater part should follow exposed segment"
    ct.commit_exposed_segment(es)

    ord_after_commit: List[ContigDescriptor] = []
    ct.traverse(generate_traverse_fn(ord_after_commit))

    assert cds == ord_after_commit, "Expose/commit should not modify contig order"

@pytest.mark.randomize(
    log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0,)
@pytest.mark.randomize(
    contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(
    contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(
    contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0, max_num=1000)
@pytest.mark.randomize(segment_start=int, min_num=-10000000, max_num=20000000)
@pytest.mark.randomize(segment_end=int, min_num=-100000000, max_num=20000000)
@pytest.mark.randomize(resolution_idx=int, min_num=0, max_num=9)
def test_expose_by_bp(
        log_resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src,
        segment_start,
        segment_end,
        resolution_idx,
):
    resolutions = [np.abs(np.int64(10**r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    ct, cds = build_tree(
        resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src
    )

    resolution: np.int64 = resolutions[resolution_idx]

    if segment_start > segment_end:
        segment_start, segment_end = segment_end, segment_start

    es: ContigTree.ExposedSegment = ct.expose_segment_by_length(segment_start, segment_end, resolution)

    def generate_traverse_fn(lst: List[ContigDescriptor]) -> Callable[[ContigTree.Node], None]:
        def traverse_fn(node: ContigTree.Node) -> None:
            lst.append(node.contig_descriptor)

        return traverse_fn

    total_length: np.int64 = sum([cd.contig_length_at_resolution[resolution] for cd in cds])

    less_size: np.int64 = es.less.get_sizes()[0][resolution] if es.less is not None else 0
    segm_size: np.int64 = es.segment.get_sizes()[0][resolution] if es.segment is not None else 0
    last_size: np.int64 = es.greater.get_sizes()[0][resolution] if es.greater is not None else 0

    assert (
            (
                    less_size + segm_size
            ) >= min(segment_end, total_length)
    ), "Segment length must be end-start+1 if segment falls into the tree"
    assert (
            less_size <= (segment_start if segment_start >= 0 else 0)
    ), "Less contigs should be in exposed segment less"
    assert (
            segm_size >=
            (
                    max(0, min(segment_end, total_length - 1))
                    - max(0, min(segment_start, total_length - 1))
                    + (1 if (segment_start < total_length and segment_end >= 0) else 0)
            )
    ), "Segment should at least have the length of query"
    assert (
            (less_size + segm_size + last_size) == total_length
    ), "All contig descriptors must fall into less, segment or greater"
    ct.commit_exposed_segment(es)

    ord_after_commit: List[ContigDescriptor] = []
    ct.traverse(generate_traverse_fn(ord_after_commit))

    assert cds == ord_after_commit, "Expose/commit should not modify contig order"


def build_tree(
        resolutions,
        contig_directions,
        contig_lengths_bp,
        contig_lengths_at_resolution_src,
) -> Tuple[ContigTree, List[ContigDescriptor]]:
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
            {res: (ContigHideType.AUTO_HIDDEN if i%2==0 else ContigHideType.FORCED_HIDDEN) if (res != np.int64(0) and contig_lengths_bp[i] < res) else (ContigHideType.AUTO_SHOWN if i%2==0 else ContigHideType.FORCED_SHOWN) for res in resolutions},
            None
        ))

    ct: ContigTree = ContigTree(np.array(resolutions))
    for i in range(0, contig_count):
        ct.insert_at_position(contig_descriptors[i], i)
    return ct, contig_descriptors
