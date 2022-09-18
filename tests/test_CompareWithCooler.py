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
hict_file = ContactMatrixFacet.get_file_descriptor(str(hict_file_path), 4)
ContactMatrixFacet.open_file(hict_file)
resolutions_hict = ContactMatrixFacet.get_resolutions_list(hict_file)
resolution_to_size_bins: Dict[np.int64, np.int64] = dict()
assert hict_file.contig_tree.root is not None, "HiCT file has no matrix inside?"
total_bp_length = hict_file.contig_tree.root.get_sizes()[0][0]
hict_file_lock: rwlock.RWLockWrite = rwlock.RWLockWrite()


def test_resolutions_match():
    assert (
        sorted(resolutions_mcool) == sorted(resolutions_hict)
    ), "Resolutions in mCool and HiCT files should match"

# NOTE: Query size is not limited so this method may fail due to the OoM


@pytest.mark.randomize(resolution=int, choices=resolutions_mcool, ncalls=len(resolutions_mcool))
@pytest.mark.randomize(start_row_incl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(start_col_incl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(end_row_excl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(end_col_excl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
def test_compare_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    end_row_excl_bp,
    end_col_excl_bp,
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = (end_row_excl_bp // resolution) % matrix_size_bins
    end_col_excl = (end_col_excl_bp // resolution) % matrix_size_bins
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    if end_row_excl - start_row_incl > 2048:
        end_row_excl = start_row_incl + \
            ((end_row_excl - start_row_incl) % 2048)
    if end_col_excl - start_col_incl > 2048:
        end_col_excl = start_col_incl + \
            ((end_col_excl - start_col_incl) % 2048)
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_wlock() as hfl:
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file, resolution, start_row_incl, start_col_incl, end_row_excl, end_col_excl, units=QueryLengthUnit.BINS, exclude_hidden_contigs=False)
    my_dense = np.pad(my_dense, [(0, end_row_excl-start_row_incl-my_dense.shape[0]), (0,
                      end_col_excl-start_col_incl-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (end_row_excl-start_row_incl,
                           end_col_excl-start_col_incl)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(end_row_excl-start_row_incl, end_col_excl-start_col_incl)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense random submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    hict_file.clear_caches(saved_blocks=True)
    gc.collect()

@pytest.mark.randomize(resolution=int, choices=resolutions_mcool, ncalls=len(resolutions_mcool))
@pytest.mark.randomize(start_row_incl=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(start_col_incl=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(end_row_excl=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(end_col_excl=int, min_num=0, max_num=total_bp_length, ncalls=5)
def test_compare_with_cooler_by_bins(
    resolution,
    start_row_incl,
    start_col_incl,
    end_row_excl,
    end_col_excl,
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl %= matrix_size_bins
    start_col_incl %= matrix_size_bins
    end_row_excl %= matrix_size_bins
    end_col_excl %= matrix_size_bins
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    if end_row_excl - start_row_incl > 2048:
        end_row_excl = start_row_incl + \
            ((end_row_excl - start_row_incl) % 2048)
    if end_col_excl - start_col_incl > 2048:
        end_col_excl = start_col_incl + \
            ((end_col_excl - start_col_incl) % 2048)
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_wlock() as hfl:
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file, resolution, start_row_incl, start_col_incl, end_row_excl, end_col_excl, units=QueryLengthUnit.BINS, exclude_hidden_contigs=False)
    my_dense = np.pad(my_dense, [(0, end_row_excl-start_row_incl-my_dense.shape[0]), (0,
                      end_col_excl-start_col_incl-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (end_row_excl-start_row_incl,
                           end_col_excl-start_col_incl)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(end_row_excl-start_row_incl, end_col_excl-start_col_incl)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense random submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    hict_file.clear_caches(saved_blocks=True)
    gc.collect()


@pytest.mark.randomize(resolution=int, choices=resolutions_mcool, ncalls=len(resolutions_mcool))
@pytest.mark.randomize(start_row_incl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(start_col_incl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(query_size=int, choices=[1, 2, 5, 10, 64, 100, 127, 512, 1000, 2560], ncalls=10)
def test_compare_square_queries_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    query_size
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = start_row_incl + query_size
    end_col_excl = start_col_incl + query_size
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_wlock() as hfl:
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file, resolution, start_row_incl, start_col_incl, end_row_excl, end_col_excl, units=QueryLengthUnit.BINS, exclude_hidden_contigs=False)
    my_dense = np.pad(my_dense, [(0, query_size-my_dense.shape[0]), (0,
                      query_size-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (query_size, query_size)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(query_size, query_size)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense square submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    hict_file.clear_caches(saved_blocks=True)
    gc.collect()

@pytest.mark.randomize(resolution=int, choices=resolutions_mcool, ncalls=len(resolutions_mcool))
@pytest.mark.randomize(start_row_incl=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(start_col_incl=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(query_size=int, choices=[1, 2, 5, 10, 64, 100, 127, 512, 1000, 2560], ncalls=10)
def test_compare_square_queries_with_cooler_by_bins(
    resolution,
    start_row_incl,
    start_col_incl,
    query_size
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl %= matrix_size_bins
    start_col_incl %= matrix_size_bins
    end_row_excl = start_row_incl + query_size
    end_col_excl = start_col_incl + query_size
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_wlock() as hfl:
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file, resolution, start_row_incl, start_col_incl, end_row_excl, end_col_excl, units=QueryLengthUnit.BINS, exclude_hidden_contigs=False)
    my_dense = np.pad(my_dense, [(0, query_size-my_dense.shape[0]), (0,
                      query_size-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (query_size, query_size)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(query_size, query_size)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense square submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    hict_file.clear_caches(saved_blocks=True)
    gc.collect()


@pytest.mark.randomize(resolution=int, choices=resolutions_mcool, ncalls=len(resolutions_mcool))
@pytest.mark.randomize(start_row_incl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(start_col_incl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(query_size_row=int, choices=[1, 2, 10, 100, 1000], ncalls=5)
@pytest.mark.randomize(query_size_col=int, choices=[1, 2, 5, 64, 127], ncalls=5)
def test_compare_rectangular_queries_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    query_size_row,
    query_size_col
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = start_row_incl + query_size_row
    end_col_excl = start_col_incl + query_size_col
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_wlock() as hfl:
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file, resolution, start_row_incl, start_col_incl, end_row_excl, end_col_excl, units=QueryLengthUnit.BINS, exclude_hidden_contigs=False)
    my_dense = np.pad(my_dense, [(0, query_size_row-my_dense.shape[0]), (0,
                      query_size_col-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (query_size_row, query_size_col)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(query_size_row, query_size_col)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense rectangular submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    hict_file.clear_caches(saved_blocks=True)
    gc.collect()


@pytest.mark.randomize(resolution=int, choices=resolutions_mcool, ncalls=len(resolutions_mcool))
@pytest.mark.randomize(start_row_incl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(start_col_incl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(end_row_excl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
@pytest.mark.randomize(end_col_excl_bp=int, min_num=0, max_num=total_bp_length, ncalls=5)
def test_hict_file_should_be_symmetric(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    end_row_excl_bp,
    end_col_excl_bp,
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = (end_row_excl_bp // resolution) % matrix_size_bins
    end_col_excl = (end_col_excl_bp // resolution) % matrix_size_bins
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    if end_row_excl - start_row_incl > 2048:
        end_row_excl = start_row_incl + \
            ((end_row_excl - start_row_incl) % 2048)
    if end_col_excl - start_col_incl > 2048:
        end_col_excl = start_col_incl + \
            ((end_col_excl - start_col_incl) % 2048)
    with hict_file_lock.gen_wlock() as hfl:
        plain_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file, resolution, start_row_incl, start_col_incl, end_row_excl, end_col_excl, units=QueryLengthUnit.BINS, exclude_hidden_contigs=False)
        transposed_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file, resolution, start_col_incl, start_row_incl, end_col_excl, end_row_excl, units=QueryLengthUnit.BINS, exclude_hidden_contigs=False)
    assert (
        np.array_equal(plain_dense, transposed_dense.T)
    ), "HiC contact matrix returned by HiCT should be symmetric"
    del plain_dense
    del transposed_dense
    hict_file.clear_caches(saved_blocks=True)
    gc.collect()
