import time
import random

import copy
from pathlib import Path
from readerwriterlock import rwlock
import cooler
from typing import Callable, Dict, List, Set, Tuple
import math

import numpy as np
import pytest
from pytest_quickcheck.generator import list_of
from hict.core.contig_tree import ContigTree
from hict.core.stripe_tree import StripeTree

from hict.core.common import ContigDescriptor, ContigDirection, ContigHideType, StripeDescriptor
from hict.api.ContactMatrixFacet import ContactMatrixFacet

random.seed(int(time.time()))

max_stripe_size: int = 256


@pytest.mark.randomize(log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0, )
@pytest.mark.randomize(resolution_index=int, min_num=0, max_num=9)
@pytest.mark.randomize(contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(contig_index=int, min_num=0, max_num=99, ncalls=15)
@pytest.mark.randomize(contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0,
    max_num=1000)
def test_reverse_single_contig(log_resolutions, resolution_index, contig_directions, contig_lengths_bp, contig_index,
        contig_lengths_at_resolution_src, ):
    resolutions = [np.abs(np.int64(10 ** r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    resolution = resolutions[resolution_index]
    ct, cds, st, sds = build_trees(resolutions, resolution, contig_directions, contig_lengths_bp,
        contig_lengths_at_resolution_src)
    stripe_count: int = len(sds)

    assert st.root is not None, "Stripe tree is empty?"

    ctgd, location_in_resolutions, location_in_resolutions_excluding_hidden, left_subsize_count = ct.get_contig_location(
        contig_index)

    cds = copy.deepcopy(cds)
    sds = copy.deepcopy(sds)

    ct.reverse_contigs_in_segment(contig_index, contig_index)

    st.reverse_direction_in_bins(1 + location_in_resolutions[resolution][0], location_in_resolutions[resolution][1])

    def generate_stripe_tree_traverse_fn(lst: List[StripeDescriptor]) -> Callable[[StripeTree.Node], None]:
        def st_traverse_fn(node: StripeTree.Node) -> None:
            lst.append(node.stripe_descriptor)

        return st_traverse_fn

    cds_after_reversal: List[ContigDescriptor] = []
    sds_after_reversal: List[StripeDescriptor] = []

    st.traverse(generate_stripe_tree_traverse_fn(sds_after_reversal))

    def generate_contig_tree_traverse_fn(lst: List[ContigDescriptor]) -> Callable[[ContigTree.Node], None]:
        def ct_traverse_fn(node: ContigTree.Node) -> None:
            lst.append(node.contig_descriptor)

        return ct_traverse_fn

    ct.traverse(generate_contig_tree_traverse_fn(cds_after_reversal))

    assert (len(cds) == len(cds_after_reversal)), "Reversal should not modify contig count"
    assert (len(sds) == len(sds_after_reversal)), "Reversal should not modify stripe count"

    if contig_index < 0 or contig_index >= len(cds):
        assert cds == cds_after_reversal, "If index is out of range, no changes should be made to contigs"
        assert sds == sds_after_reversal, "If index is out of range, no changes should be made to stripes"
    else:
        assert cds[:contig_index] == cds_after_reversal[:
                                                        contig_index], "Contigs before reversed should not be modified"
        assert cds[1 + contig_index:] == cds_after_reversal[
                                         1 + contig_index:], "Contigs before reversed should not be modified"
        assert cds[contig_index].direction != cds_after_reversal[
            contig_index].direction, "Requested contig should be reversed"
        assert (list(filter(lambda s: s.contig_descriptor.contig_id != contig_index, sds)) == list(
            filter(lambda s: s.contig_descriptor.contig_id != contig_index,
                sds_after_reversal))), "Stripes of other contigs should not be modified"
        assert all([sd.contig_descriptor.direction != cds[contig_index].direction for sd in
            filter(lambda s: s.contig_descriptor.contig_id == contig_index,
                sds_after_reversal)]), "All stripes of reversed contig should be reversed"


@pytest.mark.randomize(log_resolutions=list_of(float, items=10), min_num=1.0, max_num=9.0, )
@pytest.mark.randomize(resolution_index=int, min_num=0, max_num=9)
@pytest.mark.randomize(contig_directions=list_of(int, items=100), min_num=0, max_num=1)
@pytest.mark.randomize(contig_lengths_bp=list_of(int, items=100), min_num=0, max_num=1000000)
@pytest.mark.randomize(start_contig_id=int, min_num=0, max_num=99, ncalls=15)
@pytest.mark.randomize(end_contig_id=int, min_num=0, max_num=99, ncalls=15)
@pytest.mark.randomize(contig_lengths_at_resolution_src=list_of(list_of(int, items=10), items=100), min_num=0,
    max_num=1000)
def test_reverse_contig_segment(log_resolutions, resolution_index, contig_directions, contig_lengths_bp,
        start_contig_id, end_contig_id, contig_lengths_at_resolution_src, ):
    resolutions = [np.abs(np.int64(10 ** r)) for r in log_resolutions]
    contig_count: int = len(contig_lengths_bp)
    resolution = resolutions[resolution_index]
    ct, cds, st, sds = build_trees(resolutions, resolution, contig_directions, contig_lengths_bp,
        contig_lengths_at_resolution_src)
    stripe_count: int = len(sds)

    assert st.root is not None, "Stripe tree is empty?"

    ctgd, location_in_resolutions, location_in_resolutions_excluding_hidden, left_subsize_count = ct.get_contig_location(
        start_contig_id)

    cds = copy.deepcopy(cds)
    sds = copy.deepcopy(sds)

    cds_to_be_reversed: List[ContigDescriptor] = []

    for cd in cds:
        if len(cds_to_be_reversed) > 0 or cd.contig_id == start_contig_id:
            cds_to_be_reversed.append(cd)
        if cd.contig_id == end_contig_id:
            break

    contig_ids_to_be_reversed: Set[np.int64] = set(map(lambda cd: cd.contig_id, cds_to_be_reversed))

    ct.reverse_contigs_in_segment(start_contig_id, end_contig_id)

    st.reverse_direction_in_bins(1 + location_in_resolutions[resolution][0], location_in_resolutions[resolution][1])

    def generate_stripe_tree_traverse_fn(lst: List[StripeDescriptor]) -> Callable[[StripeTree.Node], None]:
        def st_traverse_fn(node: StripeTree.Node) -> None:
            lst.append(node.stripe_descriptor)

        return st_traverse_fn

    cds_after_reversal: List[ContigDescriptor] = []
    sds_after_reversal: List[StripeDescriptor] = []

    st.traverse(generate_stripe_tree_traverse_fn(sds_after_reversal))

    def generate_contig_tree_traverse_fn(lst: List[ContigDescriptor]) -> Callable[[ContigTree.Node], None]:
        def ct_traverse_fn(node: ContigTree.Node) -> None:
            lst.append(node.contig_descriptor)

        return ct_traverse_fn

    ct.traverse(generate_contig_tree_traverse_fn(cds_after_reversal))

    assert (len(cds) == len(cds_after_reversal)), "Reversal should not modify contig count"
    assert (len(sds) == len(sds_after_reversal)), "Reversal should not modify stripe count"

    if end_contig_id < 0 or start_contig_id >= len(cds):
        assert cds == cds_after_reversal, "If index is out of range, no changes should be made to contigs"
        assert sds == sds_after_reversal, "If index is out of range, no changes should be made to stripes"
    else:
        assert cds[:start_contig_id] == cds_after_reversal[:
                                                           start_contig_id], "Contigs before reversed segment should not be modified"
        assert cds[1 + end_contig_id:] == cds_after_reversal[
                                          1 + end_contig_id:], "Contigs before reversed segment should not be modified"
        assert all((cd.contig_id == cdr.contig_id for cd, cdr in list(zip(reversed(cds[start_contig_id:1+end_contig_id]),
                                                                          cds_after_reversal[
                                                                          start_contig_id:1+end_contig_id])))), "Requested contig segment should have contig ids reversed"
        assert all((cd.direction != cdr.direction for cd, cdr in list(zip(reversed(cds[start_contig_id:1+end_contig_id]),
                                                                          cds_after_reversal[
                                                                          start_contig_id:1+end_contig_id])))), "Requested contig segment should be reversed"
        assert (list(filter(lambda s: s.contig_descriptor.contig_id not in contig_ids_to_be_reversed, sds)) == list(
            filter(lambda s: s.contig_descriptor.contig_id not in contig_ids_to_be_reversed,
                sds_after_reversal))), "Stripes of other contigs should not be modified"
        assert all((sd.contig_descriptor.direction != cds[start_contig_id].direction for sd in
        filter(lambda s: s.contig_descriptor.contig_id in contig_ids_to_be_reversed,
            sds_after_reversal))), "All stripes of reversed contig should be reversed"


def build_trees(resolutions, resolution, contig_directions, contig_lengths_bp, contig_lengths_at_resolution_src, ) -> \
Tuple[ContigTree, List[ContigDescriptor], StripeTree, List[StripeDescriptor]]:
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
        contig_descriptors.append(
            ContigDescriptor.make_contig_descriptor(i, f"ctg-{i}", ContigDirection(np.abs(contig_directions[i]) % 2),
                contig_lengths_bp[i], contig_lengths_at_resolution[i], {
                    res: (ContigHideType.AUTO_HIDDEN if i % 2 == 0 else ContigHideType.FORCED_HIDDEN) if (
                                res != np.int64(0) and contig_lengths_bp[i] < res) else (
                        ContigHideType.AUTO_SHOWN if i % 2 == 0 else ContigHideType.FORCED_SHOWN) for res in
                    resolutions}, None))

    ct: ContigTree = ContigTree(np.array(resolutions))
    for i in range(0, contig_count):
        ct.insert_at_position(contig_descriptors[i], i)

    st: StripeTree = StripeTree(resolution)

    stripe_descriptors: List[StripeDescriptor] = []

    stripe_id: int = 0
    for i in range(0, contig_count):
        part_count: int = int(math.ceil(contig_lengths_at_resolution[i][resolution] / max_stripe_size))
        for part in range(0, max(1, part_count)):
            sd = StripeDescriptor.make_stripe_descriptor(stripe_id,
                max_stripe_size if ((1 + part) * max_stripe_size < contig_lengths_at_resolution[i][resolution]) else (
                        contig_lengths_at_resolution[i][resolution] - (part) * max_stripe_size),
                (max_stripe_size * resolution) if (
                            (1 + part) * max_stripe_size < contig_lengths_at_resolution[i][resolution]) else (
                        contig_lengths_bp[i] % (max_stripe_size * resolution)), contig_descriptors[i])
            stripe_descriptors.append(sd)
            st.insert_at_position(stripe_id, sd)
            stripe_id += 1

    return ct, contig_descriptors, st, stripe_descriptors
