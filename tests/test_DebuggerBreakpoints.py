from hict.api.ContactMatrixFacet import ContactMatrixFacet
from hict.core.common import QueryLengthUnit
import pytest
import numpy as np
from typing import Dict
import cooler
from readerwriterlock import rwlock
from pathlib import Path
import gc
import time
import random

random.seed(int(time.time()))

mcool_file_path: Path = Path(".", "..", "hict_server", "data", "zanu_male_4DN.mcool").resolve()
hict_file_path: Path = Path(".", "..", "hict_server", "data", "zanu_male_4DN.mcool.hict.hdf5").resolve()

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

resolutions_mcool = list(
    map(lambda s: int(s.replace('/resolutions/', '')), cooler.fileops.list_coolers(str(mcool_file_path))))
hict_file = ContactMatrixFacet.get_file_descriptor(str(hict_file_path), 4)
ContactMatrixFacet.open_file(hict_file)
resolutions_hict = ContactMatrixFacet.get_resolutions_list(hict_file)
resolution_to_size_bins: Dict[np.int64, np.int64] = dict()
assert hict_file.contig_tree.root is not None, "HiCT file has no matrix inside?"
total_bp_length = hict_file.contig_tree.root.get_sizes()[0][0]
hict_file_lock: rwlock.RWLockWrite = rwlock.RWLockWrite()


def test_resolutions_match():
    assert (sorted(resolutions_mcool) == sorted(resolutions_hict)), "Resolutions in mCool and HiCT files should match"


# NOTE: Query size is not limited so this method may fail due to the OoM

def compare_with_cooler(resolution, start_row_incl_bp, start_col_incl_bp, end_row_excl_bp, end_col_excl_bp, ):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(hict_file, resolution)
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = (end_row_excl_bp // resolution) % matrix_size_bins
    end_col_excl = (end_col_excl_bp // resolution) % matrix_size_bins
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    if end_row_excl - start_row_incl > 2048:
        end_row_excl = start_row_incl + ((end_row_excl - start_row_incl) % 2048)
    if end_col_excl - start_col_incl > 2048:
        end_col_excl = start_col_incl + ((end_col_excl - start_col_incl) % 2048)
    cooler_file: cooler.Cooler = cooler.Cooler("{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl, start_col_incl:end_col_excl]
    with hict_file_lock.gen_wlock() as hfl:
        my_dense = ContactMatrixFacet.get_dense_submatrix(hict_file, resolution, start_row_incl, start_col_incl,
            end_row_excl, end_col_excl, units=QueryLengthUnit.BINS, exclude_hidden_contigs=False)
    my_dense = np.pad(my_dense, [(0, end_row_excl - start_row_incl - my_dense.shape[0]),
                                 (0, end_col_excl - start_col_incl - my_dense.shape[1])], mode='constant',
                      constant_values=0)
    assert (my_dense.shape == (end_row_excl - start_row_incl,
                               end_col_excl - start_col_incl)), f"Matrix shape {my_dense.shape} should be equal to that of query: {(end_row_excl - start_row_incl, end_col_excl - start_col_incl)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)), "Dense random submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    hict_file.clear_caches(saved_blocks=True)
    gc.collect()


def test_compare_with_cooler():
    compare_with_cooler(500000, 18682465, 929736, 193416483, 64767990)
