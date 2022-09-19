import copy
import gc
import random
import time
from pathlib import Path
from typing import Dict, List, Callable

import cooler
import numpy as np
import pytest
from readerwriterlock import rwlock

from hict.api.ContactMatrixFacet import ContactMatrixFacet
from hict.core.common import ContigDescriptor, StripeDescriptor
from hict.core.contig_tree import ContigTree
from hict.core.stripe_tree import StripeTree

random.seed(int(time.time()))

mcool_file_path: Path = Path(
    ".", "..", "hict_server", "data", "zanu_male_4DN.mcool").resolve()
hict_file_path: Path = Path(
    ".", "..", "hict_server", "data", "zanu_male_4DN.mcool.hict.hdf5").resolve()

if not hict_file_path.is_file():
    pytest.exit(msg=f"Test hict file must be present for this test at {hict_file_path}")

if not mcool_file_path.is_file():
    pytest.exit(msg=f"Test mcool file must be present for this test at {mcool_file_path}")
# pytestmark = pytest.mark.skipif(
#     not hict_file_path.is_file(),
#     reason=f"Test hict file must be present for this test at {hict_file_path}"
# )
# pytestmark()
# pytestmark = pytest.mark.skipif(
#     not mcool_file_path.is_file(),
#     reason=f"Test mcool file must be present for this test at {mcool_file_path}"
# )

resolutions_mcool = list(map(lambda s: int(s.replace(
    '/resolutions/', '')), cooler.fileops.list_coolers(str(mcool_file_path))))
global_hict_file = ContactMatrixFacet.get_file_descriptor(str(hict_file_path), 1)
ContactMatrixFacet.open_file(global_hict_file)
resolutions_hict = ContactMatrixFacet.get_resolutions_list(global_hict_file)
resolution_to_size_bins: Dict[np.int64, np.int64] = dict()
assert global_hict_file.contig_tree.root is not None, "HiCT file has no matrix inside?"
contig_count: int = len(global_hict_file.contig_names)
total_bp_length = global_hict_file.contig_tree.root.get_sizes()[0][0]
hict_file_lock: rwlock.RWLockWrite = rwlock.RWLockWrite()


def generate_contig_tree_traverse_fn(lst: List[ContigDescriptor]) -> Callable[[ContigTree.Node], None]:
    def traverse_fn(node: ContigTree.Node) -> None:
        lst.append(node.contig_descriptor)

    return traverse_fn


def generate_stripe_tree_traverse_fn(lst: List[StripeDescriptor]) -> Callable[[StripeTree.Node], None]:
    def traverse_fn(node: StripeTree.Node) -> None:
        lst.append(node.stripe_descriptor)

    return traverse_fn


def check_no_scaffolds_present_in_test_file():
    cds: List[ContigDescriptor] = []
    global_hict_file.contig_tree.traverse(generate_contig_tree_traverse_fn(cds))
    if not all(map(lambda cd: cd.scaffold_id is None, cds)):
        pytest.exit(msg="Test file should not contain scaffolds")


check_no_scaffolds_present_in_test_file()


@pytest.mark.randomize(contig_id=int, min_num=0, max_num=contig_count - 1, ncalls=contig_count)
@pytest.mark.randomize(target_order=int, min_num=-10, max_num=contig_count + 10, ncalls=contig_count + 10)
@pytest.mark.randomize(stripe_tree_resolution=int, choices=resolutions_mcool)
def test_move_single_contig(
        contig_id,
        target_order,
        stripe_tree_resolution
):
    hict_file = ContactMatrixFacet.get_file_descriptor(str(hict_file_path), 1)
    ContactMatrixFacet.open_file(hict_file)
    ct_source = copy.deepcopy(hict_file.contig_tree)
    st_source = copy.deepcopy(hict_file.matrix_trees[stripe_tree_resolution])
    cds: List[ContigDescriptor] = []
    sds: List[StripeDescriptor] = []

    ct_source.traverse(generate_contig_tree_traverse_fn(cds))
    st_source.traverse(generate_stripe_tree_traverse_fn(sds))

    (
        _,
        _,
        _,
        initial_order
    ) = hict_file.get_contig_location(contig_id)

    ContactMatrixFacet.move_selection_range(hict_file, contig_id, contig_id, target_order)

    cds_after_move: List[ContigDescriptor] = []
    sds_after_move: List[StripeDescriptor] = []
    hict_file.contig_tree.traverse(generate_contig_tree_traverse_fn(cds_after_move))
    hict_file.matrix_trees[stripe_tree_resolution].traverse(generate_stripe_tree_traverse_fn(sds_after_move))

    expected_order: int = min(max(0, target_order), contig_count - 1)

    (
        cd,
        location_in_resolutions,
        location_in_resolutions_excluding_hidden,
        actual_order
    ) = hict_file.get_contig_location(contig_id)

    assert actual_order == expected_order, f"Expected to place contig with id={contig_id} at the place {expected_order} but not {actual_order}"

    assert (
            cds_after_move[expected_order] == cd
    ), "Target position should contain requested contig"

    if expected_order <= initial_order:
        assert (
                cds_after_move[:expected_order] == cds[:expected_order]
        ), "Contigs before target position should not be modified if it precedes initial position"
        assert (
                cds_after_move[1 + initial_order:] == cds[1 + initial_order:]
        ), "Contigs after initial position should not be modified if it follows target position"
        assert (
                cds_after_move[1 + expected_order:1 + initial_order] == cds[expected_order:initial_order]
        ), "Contigs between target and initial positions should be cyclically shifted right by one position"
    else:
        # initial_order < targetOrder
        assert (
                cds_after_move[:initial_order] == cds[:initial_order]
        ), "Contigs before initial position should not be modified if it precedes target position"
        assert (
                cds_after_move[1 + expected_order:] == cds[1 + expected_order:]
        ), "Contigs after target position should not be modified if it follows initial position"
        assert (
                cds_after_move[initial_order:expected_order] == cds[1 + initial_order:1 + expected_order]
        ), "Contigs between initial and target positions should be cyclically shifted left by one position"
        # 01234567
        # 012I45T7
        # 01245TI7

    ContactMatrixFacet.move_selection_range(hict_file, contig_id, contig_id, initial_order)
    (
        _,
        _,
        _,
        returned_order
    ) = hict_file.get_contig_location(contig_id)

    cds_after_return: List[ContigDescriptor] = []
    sds_after_return: List[StripeDescriptor] = []
    hict_file.contig_tree.traverse(generate_contig_tree_traverse_fn(cds_after_return))
    hict_file.matrix_trees[stripe_tree_resolution].traverse(generate_stripe_tree_traverse_fn(sds_after_return))

    assert (
            returned_order == initial_order
    ), "After moving contig back and forth, it should be at its original position"

    assert (
            cds_after_return == cds
    ), "Contig descriptors should not be modified after move and return"
    assert (
            sds_after_return == sds
    ), "Stripe descriptors should not be modified after move and return"

    hict_file.clear_caches(saved_blocks=True)
    gc.collect()
