import threading
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Iterable, Set, Union
import multiprocessing
import multiprocessing.managers
import copy

import h5py
import numpy as np
# from cachetools import LRUCache, cachedmethod
# from cachetools.keys import hashkey
from readerwriterlock import rwlock
from scipy.sparse import coo_array, csr_array, csc_array

from hict.core.AGPProcessor import *
from hict.core.FASTAProcessor import FASTAProcessor
from hict.core.common import ATUDescriptor, ATUDirection, LocationInAssembly, StripeDescriptor, ContigDescriptor, ScaffoldDescriptor, ScaffoldBorders, \
    ScaffoldDirection, FinalizeRecordType, ContigHideType, QueryLengthUnit
from hict.core.contig_tree import ContigTree
from hict.core.scaffold_holder import ScaffoldHolder
# from hict.core.stripe_tree import StripeTree
from hict.util.h5helpers import create_dataset_if_not_exists, get_attribute_value_or_create_if_not_exists, \
    create_group_if_not_exists


additional_dataset_creation_args = {
    'compression': 'lzf',
    'shuffle': True,
    'chunks': True,
}


def constrain_coordinate(x_bins: Union[np.int64, int], lower: Union[np.int64, int],
                         upper: Union[np.int64, int]) -> np.int64:
    return max(min(x_bins, upper), lower)


# BLOCK_CACHE_SIZE: int = 1024


class ChunkedFile(object):
    class FileState(Enum):
        CLOSED = 0
        OPENED = 1
        INCORRECT = 2

    def __init__(
            self,
            filepath: str,
            block_cache_size: int = 64,
            multithreading_pool_size: int = 8,
            mp_manager: Optional[multiprocessing.managers.SyncManager] = None
    ) -> None:
        super().__init__()
        self.filepath: str = filepath
        self.stripes: Dict[np.int64, List[StripeDescriptor]] = dict()
        self.atl: Dict[np.int64, List[ATUDescriptor]] = dict()
        self.contig_names: List[str] = []
        self.contig_name_to_contig_id: Dict[str, np.int64] = dict()
        self.contig_lengths_bp: Dict[np.int64, np.int64] = dict()
        self.resolutions: List[np.int64] = []
        self.contig_tree: Optional[ContigTree] = None
        self.state: ChunkedFile.FileState = ChunkedFile.FileState.CLOSED
        self.dense_submatrix_size: Dict[np.int64,
                                        np.int64] = dict()  # Resolution -> MSS
        self.block_cache_size = block_cache_size
        # self.block_cache = LRUCache(maxsize=self.block_cache_size)
        # self.block_intersection_cache = LRUCache(maxsize=self.block_cache_size)
        # self.block_cache_lock: Lock = threading.Lock()
        # self.block_intersection_cache_lock: Lock = threading.Lock()
        self.scaffold_holder: ScaffoldHolder = ScaffoldHolder()
        self.dtype: Optional[np.dtype] = None
        self.mp_manager = mp_manager
        if mp_manager is not None:
            lock_factory = mp_manager.RLock
        else:
            lock_factory = threading.RLock
        self.hdf_file_lock: rwlock.RWLockWrite = rwlock.RWLockWrite(
            lock_factory=lock_factory)
        self.opened_hdf_file: h5py.File = h5py.File(filepath, mode='r')
        self.fasta_processor: Optional[FASTAProcessor] = None
        self.fasta_file_lock: rwlock.RWLockFair = rwlock.RWLockFair(
            lock_factory=lock_factory)
        self.multithreading_pool_size = multithreading_pool_size

    def open(self):
        # NOTE: When file is opened in this method, we assert no one writes to it
        contig_id_to_length_by_resolution: Dict[np.int64,
                                                Dict[np.int64, np.int64]] = dict()
        contig_id_to_hide_type_by_resolution: Dict[np.int64,
                                                   Dict[np.int64, ContigHideType]] = dict()
        contig_id_to_direction: List[ContigDirection] = []
        contig_id_to_scaffold_id: List[Optional[np.int64]] = []
        ordered_contig_ids: np.ndarray
        with self.hdf_file_lock.gen_rlock():
            f = self.opened_hdf_file
            resolutions = np.array(
                [np.int64(sdn) for sdn in sorted(
                    filter(lambda s: s.isnumeric(), f['resolutions'].keys()))],
                dtype=np.int64
            )
            self.resolutions = resolutions
            self.dtype = f[f'resolutions/{max(resolutions)}/treap_coo/block_vals'].dtype

            for resolution in resolutions:
                (
                    self.stripes[resolution],
                    self.dense_submatrix_size[resolution]
                ) = self.read_stripe_data(f, resolution)

            self.atl = self.read_atl(f)

            (
                contig_id_to_contig_length_bp,
                resolution_to_contig_length_bins,
                resolution_to_contig_hide_type,
                contig_id_to_atus,
                contig_names
            ) = self.read_contig_data(f)
            self.contig_names = contig_names
            for contig_id, contig_name in enumerate(contig_names):
                self.contig_name_to_contig_id[contig_name] = contig_id

            contig_count: np.int64 = len(contig_names)
            for contig_id in range(0, contig_count):
                contig_id_to_length_by_resolution[contig_id] = dict()
                contig_id_to_hide_type_by_resolution[contig_id] = dict()

            for resolution in resolutions:
                contig_id_to_contig_length_bins_at_resolution = resolution_to_contig_length_bins[
                    resolution]
                contig_id_to_contig_hide_type_at_resolution = resolution_to_contig_hide_type[
                    resolution]
                for contig_id, (
                        contig_length_bins_at_resolution,
                        contig_hide_type_at_resolution
                ) in enumerate(zip(
                    contig_id_to_contig_length_bins_at_resolution,
                    contig_id_to_contig_hide_type_at_resolution
                )):
                    contig_id_to_length_by_resolution[contig_id][resolution] = (
                        contig_length_bins_at_resolution
                    )
                    contig_id_to_hide_type_by_resolution[contig_id][resolution] = ContigHideType(
                        contig_hide_type_at_resolution
                    )

            contig_info_group: h5py.Group = f['/contig_info/']
            ordered_contig_ids: h5py.Dataset = contig_info_group['ordered_contig_ids']
            contig_direction_ds: h5py.Dataset = contig_info_group['contig_direction']
            contig_scaffold_ids: h5py.Dataset = contig_info_group['contig_scaffold_id']

            for (
                    contig_direction,
                    contig_scaff_id
            ) in zip(
                contig_direction_ds,
                contig_scaffold_ids
            ):
                contig_id_to_direction.append(
                    ContigDirection(contig_direction))
                contig_id_to_scaffold_id.append(
                    contig_scaff_id if contig_scaff_id >= 0 else None)

            self.contig_tree = ContigTree(self.resolutions)

            contig_id_to_contig_descriptor: List[ContigDescriptor] = []

            for (
                    contig_id,
                    resolution_to_contig_length
            ) in contig_id_to_length_by_resolution.items():
                contig_presence_at_resolution: Dict[
                    np.int64,
                    ContigHideType
                ] = contig_id_to_hide_type_by_resolution[contig_id]
                contig_presence_at_resolution[0] = ContigHideType.FORCED_SHOWN

                # Hide small contigs only at zoomed resolutions
                for res in resolutions[1:]:
                    if contig_id_to_contig_length_bp[contig_id] < res:
                        contig_presence_at_resolution[res] = ContigHideType.AUTO_HIDDEN
                contig_descriptor: ContigDescriptor = ContigDescriptor.make_contig_descriptor(
                    contig_id=contig_id,
                    contig_name=contig_names[contig_id],
                    # direction=contig_id_to_direction[contig_id],
                    contig_length_bp=contig_id_to_contig_length_bp[contig_id],
                    contig_length_at_resolution=resolution_to_contig_length,
                    contig_presence_in_resolution=contig_presence_at_resolution,
                    # {resolution: list(map(lambda ati: self.atl[resolution][ati], contig_id_to_atus[contig_id][resolution])) for resolution in resolutions},
                    atus=contig_id_to_atus[contig_id],
                    scaffold_id=contig_id_to_scaffold_id[contig_id]
                )
                contig_id_to_contig_descriptor.append(contig_descriptor)

            for contig_id in ordered_contig_ids:
                contig_descriptor = contig_id_to_contig_descriptor[contig_id]
                self.contig_tree.insert_at_position(
                    contig_descriptor,
                    self.contig_tree.get_node_count(),
                    direction=contig_id_to_direction[contig_id],
                    update_tree=False
                )
            self.contig_tree.update_tree()
            self.restore_scaffolds(f)

        self.state = ChunkedFile.FileState.OPENED

    def clear_caches(self, saved_blocks: bool = False):
        return
        # if saved_blocks:
        #     if self.load_saved_dense_block.cache_lock is not None:
        #         with self.load_saved_dense_block.cache_lock(self):
        #             self.load_saved_dense_block.cache(self).clear()
        # if self.get_block_intersection_as_dense_matrix.cache_lock is not None:
        #     with self.get_block_intersection_as_dense_matrix.cache_lock(self):
        #         self.get_block_intersection_as_dense_matrix.cache(self).clear()

    def restore_scaffolds(self, f: h5py.File):
        if 'scaffold_info' not in f['/'].keys():
            # No scaffolds are present
            return
        scaffold_info_group: h5py.Group = f['/scaffold_info']
        scaffold_name_ds: h5py.Dataset = scaffold_info_group['scaffold_name']

        scaffold_start_ds: h5py.Dataset = scaffold_info_group['scaffold_start']
        scaffold_end_ds: h5py.Dataset = scaffold_info_group['scaffold_end']

        scaffold_direction_ds: h5py.Dataset = scaffold_info_group['scaffold_direction']
        scaffold_spacer_ds: h5py.Dataset = scaffold_info_group['scaffold_spacer']

        for scaffold_id, (
                scaffold_name,
                scaffold_start_contig_id,
                scaffold_end_contig_id,
                scaffold_direction,
                scaffold_spacer
        ) in enumerate(zip(
            scaffold_name_ds,
            scaffold_start_ds,
            scaffold_end_ds,
            scaffold_direction_ds,
            scaffold_spacer_ds,
        )):
            assert (
                (scaffold_end_contig_id == -1) if (scaffold_start_contig_id == -
                                                   1) else (scaffold_end_contig_id != -1)
            ), "Scaffold borders are existent/nonexistent separately??"
            scaffold_descriptor: ScaffoldDescriptor = ScaffoldDescriptor(
                scaffold_id=scaffold_id,
                scaffold_name=bytes(scaffold_name).decode('utf-8'),
                scaffold_borders=(
                    ScaffoldBorders(
                        scaffold_start_contig_id,
                        scaffold_end_contig_id
                    ) if scaffold_start_contig_id != -1 else None
                ),
                scaffold_direction=ScaffoldDirection(scaffold_direction),
                spacer_length=scaffold_spacer,
            )
            self.scaffold_holder.insert_saved_scaffold__(scaffold_descriptor)

    def read_atl(
        self,
        f: h5py.File
    ) -> Dict[np.int64, List[ATUDescriptor]]:
        resolution_atus: Dict[np.int64, ATUDescriptor] = dict()

        for resolution in self.resolutions:
            atl_group: h5py.Group = f[f'/resolutions/{resolution}/atl']
            basis_atu: h5py.Dataset = atl_group['basis_atu']

            atus = [
                ATUDescriptor.make_atu_descriptor(
                    stripe_descriptor=self.stripes[resolution][row[0]],
                    start_index_in_stripe_incl=row[1],
                    end_index_in_stripe_excl=row[2],
                    direction=ATUDirection(row[3])
                ) for row in basis_atu
            ]

            resolution_atus[resolution] = atus

        return resolution_atus

    def read_contig_data(
            self,
            f: h5py.File
    ) -> Tuple[
        np.ndarray,
        Dict[np.int64, np.ndarray],
        Dict[np.int64, np.ndarray],
        List[Dict[np.int64, List[ATUDescriptor]]],
        List[str]
    ]:
        contig_info_group: h5py.Group = f['/contig_info/']
        contig_names_ds: h5py.Dataset = contig_info_group['contig_name']
        contig_lengths_bp: h5py.Dataset = contig_info_group['contig_length_bp']

        contig_count: np.int64 = len(contig_names_ds)

        assert len(
            contig_lengths_bp) == contig_count, "Different contig count in different datasets??"

        # Resolution -> [ContigId -> ContigLengthBp]
        resolution_to_contig_length_bins: Dict[np.int64, np.ndarray] = dict()
        # Resolution -> [ContigId -> ContigHideType]
        resolution_to_contig_hide_type: Dict[np.int64, np.ndarray] = dict()
        # resolution_to_contig_atus: Dict[np.int64, List[List[ATUDescriptor]]] = dict()
        contig_id_to_atus: List[Dict[np.int64, List[ATUDescriptor]]] = [
            {resolution: [] for resolution in self.resolutions} for _ in range(contig_count)]
        for resolution in self.resolutions:
            contigs_group: h5py.Group = f[f'/resolutions/{resolution}/contigs/']
            contig_length_bins_ds: h5py.Dataset = contigs_group['contig_length_bins']
            contig_hide_type_ds: h5py.Dataset = contigs_group['contig_hide_type']
            contig_atus: h5py.Dataset = contigs_group['atl']

            assert len(
                contig_length_bins_ds) == contig_count, "Different contig count in different datasets??"

            for contig_id, basis_atu_id in contig_atus:
                contig_id_to_atus[contig_id][resolution].append(
                    self.atl[resolution][basis_atu_id])

            resolution_to_contig_length_bins[resolution] = np.array(
                contig_length_bins_ds[:].astype(np.int64),
                dtype=np.int64
            )

            resolution_to_contig_hide_type[resolution] = np.array(
                contig_hide_type_ds[:].astype(np.int64),
                dtype=np.int8
            )

        contig_id_to_contig_length_bp: np.ndarray = np.array(
            contig_lengths_bp[:].astype(np.int64), dtype=np.int64)
        contig_names: List[str] = [bytes(contig_name).decode(
            'utf-8') for contig_name in contig_names_ds]

        return (
            contig_id_to_contig_length_bp,
            resolution_to_contig_length_bins,
            resolution_to_contig_hide_type,
            contig_id_to_atus,
            contig_names
        )

    def read_stripe_data(
            self,
            f: h5py.File,
            resolution: np.int64
    ) -> Tuple[
        List[StripeDescriptor],
        np.int64
    ]:
        stripes_group: h5py.Group = f[f'/resolutions/{resolution}/stripes']
        stripe_lengths_bins: h5py.Dataset = stripes_group['stripe_length_bins']
        stripes_bin_weights: Optional[h5py.Dataset] = (
            stripes_group['stripes_bin_weights']
        ) if 'stripes_bin_weights' in stripes_group.keys() else None

        stripes: List[StripeDescriptor] = [
            StripeDescriptor.make_stripe_descriptor(
                stripe_id,
                stripe_length_bins,
                np.array(
                    np.nan_to_num(
                        stripes_bin_weights[stripe_id, :stripe_length_bins], copy=False),
                    dtype=np.float64
                ) if stripes_bin_weights is not None else np.ones(stripe_length_bins, dtype=np.float64)
            ) for (
                stripe_id, stripe_length_bins
            ) in enumerate(stripe_lengths_bins)
        ]

        dense_submatrix_size: np.int64 = max(stripe_lengths_bins)

        return stripes, dense_submatrix_size

    def sparse_to_dense(self, sparse_mx: Union[coo_array, csr_array, csc_array]) -> np.ndarray:
        return sparse_mx.todense()

    def process_flips(
        self,
        mx_as_array: np.ndarray,
        row_atu: ATUDescriptor,
        col_atu: ATUDescriptor
    ) -> np.ndarray:
        if row_atu.direction == ATUDirection.REVERSED:
            mx_as_array = np.flip(mx_as_array, axis=0)
        if col_atu.direction == ATUDirection.REVERSED:
            mx_as_array = np.flip(mx_as_array, axis=1)
        return mx_as_array

    def get_stripe_intersection_for_atus_as_raw_dense_matrix(
            self,
            resolution: np.int64,
            row_atu: ATUDescriptor,
            col_atu: ATUDescriptor
    ) -> np.ndarray:
        row_stripe: StripeDescriptor = row_atu.stripe_descriptor
        col_stripe: StripeDescriptor = col_atu.stripe_descriptor
        needs_transpose: bool = False
        if row_stripe.stripe_id > col_stripe.stripe_id:
            row_stripe, col_stripe = col_stripe, row_stripe
            needs_transpose = True

        mx_as_array: np.ndarray
        is_empty: bool

        r: np.int64 = row_stripe.stripe_id
        c: np.int64 = col_stripe.stripe_id

        with self.hdf_file_lock.gen_rlock():
            blocks_dir: h5py.Group = self.opened_hdf_file[
                f'/resolutions/{resolution}/treap_coo']
            stripes_count: np.int64 = blocks_dir.attrs['stripes_count']
            block_index_in_datasets: np.int64 = r * stripes_count + c

            block_lengths: h5py.Dataset = blocks_dir['block_length']
            block_length = block_lengths[block_index_in_datasets]
            is_empty: bool = (block_length == 0)

            if is_empty:
                mx_as_array = np.zeros(
                    shape=(
                        row_atu.end_index_in_stripe_excl - row_atu.start_index_in_stripe_incl,
                        col_atu.end_index_in_stripe_excl - col_atu.start_index_in_stripe_incl
                    ),
                    dtype=self.dtype
                )
            else:
                block_offsets: h5py.Dataset = blocks_dir['block_offset']
                block_offset = block_offsets[block_index_in_datasets]
                is_dense: bool = (block_offset < 0)

                if is_dense:
                    dense_blocks: h5py.Dataset = blocks_dir['dense_blocks']
                    index_in_dense_blocks: np.int64 = -(block_offset + 1)
                    mx_as_array = dense_blocks[index_in_dense_blocks, 0, :, :]
                else:
                    block_vals: h5py.Dataset = blocks_dir['block_vals']
                    block_finish = block_offset + block_length
                    block_rows: h5py.Dataset = blocks_dir['block_rows']
                    block_cols: h5py.Dataset = blocks_dir['block_cols']
                    mx = coo_array(
                        (
                            block_vals[block_offset:block_finish],
                            (
                                block_rows[block_offset:block_finish],
                                block_cols[block_offset:block_finish]
                            )
                        ),
                        shape=(row_stripe.stripe_length_bins,
                               col_stripe.stripe_length_bins)
                    )
                    mx_as_array = self.sparse_to_dense(mx)

                if row_atu.stripe_descriptor.stripe_id == col_atu.stripe_descriptor.stripe_id:
                    assert (
                        row_atu.stripe_descriptor == col_atu.stripe_descriptor
                    ), "Fetched stripe descriptors have the same ids, but are not equal??"
                    mx_as_array = np.where(
                        mx_as_array, mx_as_array, mx_as_array.T)

                if needs_transpose:
                    mx_as_array = mx_as_array.T

                mx_as_array = mx_as_array[
                    row_atu.start_index_in_stripe_incl:row_atu.end_index_in_stripe_excl,
                    col_atu.start_index_in_stripe_incl:col_atu.end_index_in_stripe_excl,
                ]

                mx_as_array = self.process_flips(mx_as_array, row_atu, col_atu)

        return mx_as_array

    def get_submatrix(
            self,
            resolution: np.int64,
            start_row_incl: np.int64,
            start_col_incl: np.int64,
            end_row_excl: np.int64,
            end_col_excl: np.int64,
            exclude_hidden_contigs: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        total_assembly_length = self.contig_tree.get_sizes(
        )[2 if exclude_hidden_contigs else 0][resolution]

        start_row_incl = constrain_coordinate(
            start_row_incl, 0, total_assembly_length)
        end_row_excl = constrain_coordinate(
            end_row_excl, 0, total_assembly_length)
        start_col_incl = constrain_coordinate(
            start_col_incl, 0, total_assembly_length)
        end_col_excl = constrain_coordinate(
            end_col_excl, 0, total_assembly_length)

        row_atus: List[ATUDescriptor] = self.get_atus_for_range(
            resolution,
            start_row_incl,
            end_row_excl,
            exclude_hidden_contigs
        )
        col_atus: List[ATUDescriptor] = self.get_atus_for_range(
            resolution,
            start_col_incl,
            end_col_excl,
            exclude_hidden_contigs
        )

        query_rows_count = end_row_excl - start_row_incl
        query_cols_count = end_col_excl - start_col_incl

        if start_row_incl < end_row_excl and 0 <= start_row_incl < total_assembly_length:
            assert (
                len(row_atus) > 0
            ), "Query is correct but no rows were found??"

        if start_col_incl < end_col_excl and 0 <= start_col_incl < total_assembly_length:
            assert (
                len(col_atus) > 0
            ), "Query is correct but no columns were found??"

        row_matrices: List[np.ndarray] = []
        row_subweights: List[np.ndarray] = []

        row_subtotals: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for row_atu in row_atus:
            def load_intersection(col_atu: ATUDescriptor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                return self.get_atu_intersection(
                    resolution=resolution,
                    row_atu=row_atu,
                    col_atu=col_atu
                )
            # with Pool(processes=self.multithreading_pool_size) as P:
                # row_subtotals = P.map(load_intersection, col_atus)
            row_subtotals = list(map(load_intersection, col_atus))
            row_submatrices: List[np.ndarray] = [t[0] for t in row_subtotals]
            if len(col_atus) > 0:
                assert (
                    len(row_subtotals) > 0
                ), "There were ATUs but no intersection??"
                assert all(
                    (
                        sbm.shape[0] == row_submatrices[0].shape[0]
                        for sbm in row_submatrices
                    )
                ), "Not all submatrices in row have the same row count??"
                assert (
                    row_submatrices[0].shape[0] == (
                        row_atu.end_index_in_stripe_excl - row_atu.start_index_in_stripe_incl
                    )
                ), "Row height is not equal to what ATU describes??"
                row = (
                    np.hstack(row_submatrices)
                )
                row_subweights.append(row_subtotals[0][1])
            else:
                assert (
                    query_cols_count <= 0
                ), "No column ATUs are present, but query is non-trivial for columns??"
                row = np.zeros(shape=(
                    row_atu.end_index_in_stripe_excl - row_atu.start_index_in_stripe_incl, 0))
            row_matrices.append(row)

        if len(row_subweights) > 0:
            row_weights = np.hstack(row_subweights)
        else:
            row_weights = np.ones(shape=max(0, query_rows_count))

        col_subweights = [t[2] for t in row_subtotals]
        if len(col_subweights) > 0:
            col_weights = np.hstack(col_subweights)
        else:
            col_weights = np.ones(shape=max(0, query_cols_count))

        if query_rows_count > 0 and query_cols_count > 0:
            result = np.vstack(row_matrices)
            assert (
                len(row_subweights) > 0
            ), "No row weights were fetched, but query is non-trivial for rows??"
            assert (
                len(col_subweights) > 0
            ), "No column weights were fetched, but query is non-trivial for columns??"
        else:
            result = np.zeros(
                shape=(max(0, query_rows_count), max(0, query_cols_count)))

        assert (
            result.shape[0] == (end_row_excl-start_row_incl)
        ), "Row count is not as queried??"

        assert (
            result.shape[1] == (end_col_excl-start_col_incl)
        ), "Column count is not as queried??"

        assert (
            len(row_weights) == (end_row_excl-start_row_incl)
        ), "Row weights count is not as queried??"

        assert (
            len(col_weights) == (end_col_excl-start_col_incl)
        ), "Column weights count is not as queried??"

        return result, row_weights, col_weights

    def get_atu_intersection(
        self,
        resolution: np.int64,
        row_atu: ATUDescriptor,
        col_atu: ATUDescriptor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        atu_intersection_dense: np.ndarray = self.get_stripe_intersection_for_atus_as_raw_dense_matrix(
            resolution,
            row_atu,
            col_atu
        )

        row_weights = row_atu.stripe_descriptor.bin_weights[
            row_atu.start_index_in_stripe_incl:row_atu.end_index_in_stripe_excl]
        col_weights = col_atu.stripe_descriptor.bin_weights[
            col_atu.start_index_in_stripe_incl:col_atu.end_index_in_stripe_excl]
        if row_atu.direction == ATUDirection.REVERSED:
            row_weights = np.flip(row_weights)
        if col_atu.direction == ATUDirection.REVERSED:
            col_weights = np.flip(col_weights)

        return atu_intersection_dense, row_weights, col_weights

    def get_atus_for_range(
        self,
        resolution: np.int64,
        start_px_incl: np.int64,
        end_px_excl: np.int64,
        exclude_hidden_contigs: bool,
    ) -> List[ATUDescriptor]:
        total_assembly_length = self.contig_tree.get_sizes(
        )[2 if exclude_hidden_contigs else 0][resolution]
        start_px_incl = constrain_coordinate(
            start_px_incl, 0, total_assembly_length)
        end_px_excl = constrain_coordinate(
            end_px_excl, 0, total_assembly_length)

        es: ContigTree.ExposedSegment = self.contig_tree.expose_segment(
            resolution,
            1+start_px_incl,
            end_px_excl,
            units=QueryLengthUnit.PIXELS if exclude_hidden_contigs else QueryLengthUnit.BINS
        )

        result_atus: List[ATUDescriptor]

        query_length: np.int64 = end_px_excl - start_px_incl
        if query_length <= 0:
            return []

        if es.segment is None:
            assert query_length <= 0, "Query is not zero-length, but no ATUs were found?"
            result_atus = []
        else:
            # TODO: maybe no update_sizes
            segment_size = es.segment.get_sizes(
            )[2 if exclude_hidden_contigs else 0][resolution]
            less_size: np.int64
            if es.less is not None:
                less_size = es.less.get_sizes(
                )[2 if exclude_hidden_contigs else 0][resolution]
            else:
                less_size = np.int64(0)

            delta_px_between_segment_first_contig_start_and_query_start: np.int64 = start_px_incl - less_size
            assert delta_px_between_segment_first_contig_start_and_query_start >= 0

            total_segment_length_px: np.int64 = es.segment.get_sizes(
            )[2 if exclude_hidden_contigs else 0][resolution]

            atus: List[ATUDescriptor] = []

            def traverse_fn(node: ContigTree.Node) -> None:
                contig_atus = node.contig_descriptor.atus[resolution]
                contig_direction = node.true_direction()
                if contig_direction == ContigDirection.REVERSED:
                    contig_atus = reversed(contig_atus)
                atus.extend(contig_atus)

            # TODO: maybe no need in push
            ContigTree.traverse_nodes_at_resolution(
                es.segment,
                resolution,
                exclude_hidden_contigs,
                traverse_fn
            )

            all_atus_debug = copy.deepcopy(atus)

            total_exposed_atu_length = sum(
                map(
                    lambda atu: atu.end_index_in_stripe_excl -
                    atu.start_index_in_stripe_incl,
                    atus
                )
            )

            assert (
                total_exposed_atu_length == total_segment_length_px
            ), "ATUs total length is not equal to exposed segment length??"

            # TODO: maybe no push is needed
            first_contig_node_in_segment: Optional[ContigTree.Node] = es.segment.leftmost(
            )

            assert first_contig_node_in_segment is not None, "Segment is not empty but has no leftmost contig??"

            first_contig_in_segment: ContigDescriptor = first_contig_node_in_segment.contig_descriptor

            index_of_atu_containing_start: np.int64 = np.searchsorted(
                first_contig_in_segment.atu_prefix_sum_length_bins[resolution],
                delta_px_between_segment_first_contig_start_and_query_start,
                side='right'
            )

            assert (
                index_of_atu_containing_start < len(
                    first_contig_in_segment.atu_prefix_sum_length_bins[resolution])
            ), "Start of query does not fall into exposed leftmost contig??"

            length_of_atus_before_one_containing_start_px: np.int64 = (
                first_contig_in_segment.atu_prefix_sum_length_bins[resolution][
                    index_of_atu_containing_start-1
                ] if index_of_atu_containing_start > 0 else np.int64(0)
            )

            old_first_atu = atus[index_of_atu_containing_start]
            new_first_atu: ATUDescriptor = old_first_atu.clone()
            new_first_atu.start_index_in_stripe_incl += (
                delta_px_between_segment_first_contig_start_and_query_start -
                length_of_atus_before_one_containing_start_px
            )

            assert (
                0 <= new_first_atu.start_index_in_stripe_incl < new_first_atu.stripe_descriptor.stripe_length_bins
            ), "Incorrect first ATU left border??"

            atus[index_of_atu_containing_start] = new_first_atu
            atus = atus[index_of_atu_containing_start:]

            delta_between_right_px_and_exposed_segment: np.int64 = end_px_excl - \
                (less_size + segment_size)
            # TODO: maybe no push
            last_contig_node = es.segment.rightmost()
            reversed_last_contig_atus_prefix_sum = last_contig_node.contig_descriptor.atu_prefix_sum_length_bins[
                resolution].copy()
            if last_contig_node.direction == ContigDirection.FORWARD:
                reversed_last_contig_atus_prefix_sum[:-1] = reversed_last_contig_atus_prefix_sum[-1] - np.flip(
                    reversed_last_contig_atus_prefix_sum)[1:]

            right_offset_atus: np.int64 = np.searchsorted(
                reversed_last_contig_atus_prefix_sum,
                -delta_between_right_px_and_exposed_segment,
                side='right'
            )

            deleted_atus_length: np.int64 = np.int64(0)
            if right_offset_atus > 0:
                atus = atus[:-right_offset_atus]
                deleted_atus_length = reversed_last_contig_atus_prefix_sum[right_offset_atus-1]

            old_last_atu = atus[-1]
            new_last_atu = old_last_atu.clone()
            new_last_atu.end_index_in_stripe_excl += (
                deleted_atus_length + delta_between_right_px_and_exposed_segment)
            assert (
                new_last_atu.stripe_descriptor.stripe_length_bins >= new_last_atu.end_index_in_stripe_excl > new_last_atu.start_index_in_stripe_incl
            ), "Incorrect ATU right border??"
            atus[-1] = new_last_atu

            assert all(map(
                lambda atu: atu.start_index_in_stripe_incl < atu.end_index_in_stripe_excl,
                atus
            )), "Incorrect ATUs before reduce??"

            total_atu_length = sum(
                map(
                    lambda atu: atu.end_index_in_stripe_excl -
                    atu.start_index_in_stripe_incl, atus
                )
            )

            assert (
                total_atu_length
                == (
                    min(end_px_excl, total_assembly_length) -
                    start_px_incl
                )
            ), "ATUs total length is not equal to the requested query??"

            result_atus = ATUDescriptor.reduce(atus)

            assert all(map(
                lambda atu: atu.start_index_in_stripe_incl < atu.end_index_in_stripe_excl,
                result_atus
            )), "Incorrect ATUs after reduce??"

        assert (
            (len(result_atus) <= 0) == (start_px_incl >= end_px_excl)
        ), "No row ATUs were fetched but query is correct??"

        total_result_atu_length = sum(
            map(
                lambda atu: atu.end_index_in_stripe_excl -
                atu.start_index_in_stripe_incl, result_atus
            )
        )

        assert (
            total_result_atu_length
            == (
                min(end_px_excl, total_assembly_length) -
                start_px_incl
            )
        ), "Resulting ATUs total length is not equal to the requested query??"

        return result_atus

    def reverse_selection_range(self, queried_start_contig_id: np.int64, queried_end_contig_id: np.int64) -> None:
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"

        queried_start_contig_order: np.int64 = self.contig_tree.get_contig_order(
            queried_start_contig_id)[1]
        queried_end_contig_order: np.int64 = self.contig_tree.get_contig_order(
            queried_end_contig_id)[1]

        if queried_end_contig_order < queried_start_contig_order:
            (queried_start_contig_id, queried_start_contig_order, queried_end_contig_id, queried_end_contig_order) = (
                queried_end_contig_id, queried_end_contig_order, queried_start_contig_id, queried_start_contig_order)

        queried_start_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_start_contig_id)
        queried_end_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_end_contig_id)

        queried_start_contig_scaffold_id = queried_start_node.contig_descriptor.scaffold_id
        queried_end_contig_scaffold_id = queried_end_node.contig_descriptor.scaffold_id

        start_contig_id: np.int64 = (
            queried_start_contig_id
            if queried_start_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(
                queried_start_contig_scaffold_id).scaffold_borders.start_contig_id
        )
        end_contig_id: np.int64 = (
            queried_end_contig_id
            if queried_end_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(queried_end_contig_scaffold_id).scaffold_borders.end_contig_id
        )

        (
            _,
            borders_bins_start,
            _,
            start_index
        ) = self.contig_tree.get_contig_location(start_contig_id)
        (
            _,
            borders_bins_end,
            _,
            end_index
        ) = self.contig_tree.get_contig_location(end_contig_id)

        if end_index < start_index:
            raise Exception(
                f"After selection was extended, its end contig with ID={end_contig_id} and order {end_index} precedes start contig with ID={start_contig_id} and order={start_index}")

        ct_exposed_segment = self.contig_tree.expose_segment_by_count(
            start_index, end_index)
        scaffold_ids_in_subtree: Set[np.int64] = set()
        if ct_exposed_segment.segment is not None:
            ct_exposed_segment.segment.reverse_subtree()

            def traverse_fn(n: ContigTree.Node):
                if n.contig_descriptor.scaffold_id is not None:
                    scaffold_ids_in_subtree.add(
                        n.contig_descriptor.scaffold_id)

            ContigTree.traverse_node(ct_exposed_segment.segment, traverse_fn)
        self.contig_tree.commit_exposed_segment(ct_exposed_segment)

        for scaffold_id in scaffold_ids_in_subtree:
            self.scaffold_holder.reverse_scaffold(scaffold_id)

        for resolution in self.resolutions:
            mt = self.matrix_trees[resolution]
            (start_bins, end_bins) = (
                borders_bins_start[resolution][0], borders_bins_end[resolution][1])
            mt.reverse_direction_in_bins(1 + start_bins, end_bins)
        self.clear_caches()

    def move_selection_range(self, queried_start_contig_id: np.int64, queried_end_contig_id: np.int64,
                             target_start_order: np.int64) -> None:
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"

        queried_start_contig_order: np.int64 = self.contig_tree.get_contig_order(
            queried_start_contig_id)[1]
        queried_end_contig_order: np.int64 = self.contig_tree.get_contig_order(
            queried_end_contig_id)[1]

        if queried_end_contig_order < queried_start_contig_order:
            (queried_start_contig_id, queried_start_contig_order, queried_end_contig_id, queried_end_contig_order) = (
                queried_end_contig_id, queried_end_contig_order, queried_start_contig_id, queried_start_contig_order)

        queried_start_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_start_contig_id)
        queried_end_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_end_contig_id)

        queried_start_contig_scaffold_id = queried_start_node.contig_descriptor.scaffold_id
        queried_end_contig_scaffold_id = queried_end_node.contig_descriptor.scaffold_id

        start_contig_id: np.int64 = (
            queried_start_contig_id
            if queried_start_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(
                queried_start_contig_scaffold_id).scaffold_borders.start_contig_id
        )
        end_contig_id: np.int64 = (
            queried_end_contig_id
            if queried_end_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(queried_end_contig_scaffold_id).scaffold_borders.end_contig_id
        )

        (
            _,
            borders_bins_start,
            _,
            start_index
        ) = self.contig_tree.get_contig_location(start_contig_id)
        (
            _,
            borders_bins_end,
            _,
            end_index
        ) = self.contig_tree.get_contig_location(end_contig_id)

        if end_index < start_index:
            raise Exception(
                f"After selection was extended, its end contig with ID={end_contig_id} and order {end_index} precedes start contig with ID={start_contig_id} and order={start_index}")

        (previous_contig_scaffold,
         target_contig_scaffold,
         next_contig_scaffold,
         scaffold_starts_at_target,
         internal_scaffold_contig,
         scaffold_ends_at_target) = self.check_scaffold_borders_at_position(target_start_order)

        target_scaffold_id: Optional[np.int64] = None

        if internal_scaffold_contig:
            target_scaffold_id = target_contig_scaffold.scaffold_id

        ct_exposed_segment = self.contig_tree.expose_segment_by_count(
            start_index, end_index)
        if ct_exposed_segment.segment is not None:
            if target_scaffold_id is not None:
                ct_exposed_segment.segment.contig_descriptor.scaffold_id = target_scaffold_id
                ct_exposed_segment.segment.needs_updating_scaffold_id_in_subtree = True
        tmp_tree: Optional[ContigTree.Node] = self.contig_tree.merge_nodes(
            ct_exposed_segment.less, ct_exposed_segment.greater)
        l, r = self.contig_tree.split_node_by_count(
            tmp_tree, target_start_order)
        leftLength: Dict[np.int64, np.int64]
        if l is not None:
            leftLength = l.get_sizes()[0]
        else:
            leftLength = dict().fromkeys(self.resolutions, 0)
        self.contig_tree.commit_exposed_segment(
            ContigTree.ExposedSegment(l, ct_exposed_segment.segment, r))

        for resolution in self.resolutions:
            mt = self.matrix_trees[resolution]
            (start_bins, end_bins) = (
                borders_bins_start[resolution][0], borders_bins_end[resolution][1])
            mt.move_stripes(1+start_bins, end_bins, leftLength[resolution])
        self.clear_caches()

    def reverse_selection_range_bp(self, queried_start_bp: np.int64, queried_end_bp: np.int64) -> None:
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"

        assert (
            queried_start_bp < queried_end_bp
        ), "Left contig border should be less than right"

        with self.contig_tree.root_lock.gen_wlock():
            reqested_segment: ContigTree.ExposedSegment = self.contig_tree.expose_segment(
                resolution=resolution,
                start=queried_start_bp,
                end=queried_end_bp-1,
                units=QueryLengthUnit.BASE_PAIRS
            )
            
            if reqested_segment.segment is None:
                return
            
            left_ctg = reqested_segment.segment.leftmost()
            right_ctg = reqested_segment.segment.rightmost()
            
            assert left_ctg is not None
            assert right_ctg is not None
            
            queried_start_contig_scaffold_id = left_ctg.scaffold_id
            queried_end_contig_scaffold_id = right_ctg.scaffold_id
            
            
            start_contig: ContigTree.Node = (
                left_ctg
                if queried_start_contig_scaffold_id is None
                else self.contig_tree.contig_id_to_node_in_tree[self.scaffold_holder.get_scaffold_by_id(
                    queried_start_contig_scaffold_id).scaffold_borders.start_contig_id]
            )
            
            end_contig: ContigTree.Node = (
                right_ctg
                if queried_end_contig_scaffold_id is None
                else self.contig_tree.contig_id_to_node_in_tree[self.scaffold_holder.get_scaffold_by_id(
                    queried_end_contig_scaffold_id).scaffold_borders.end_contig_id]
            )
                        
            start_contig_location: LocationInAssembly = self.contig_tree.contig_id_to_location_in_assembly[
                start_contig.contig_descriptor.contig_id
            ]
            
            end_contig_location: LocationInAssembly = self.contig_tree.contig_id_to_location_in_assembly[
                end_contig.contig_descriptor.contig_id
            ]

        (
            _,
            borders_bins_start,
            _,
            start_index
        ) = self.contig_tree.get_contig_location(start_contig_id)
        (
            _,
            borders_bins_end,
            _,
            end_index
        ) = self.contig_tree.get_contig_location(end_contig_id)

        if end_index < start_index:
            raise Exception(
                f"After selection was extended, its end contig with ID={end_contig_id} and order {end_index} precedes start contig with ID={start_contig_id} and order={start_index}")

        ct_exposed_segment = self.contig_tree.expose_segment_by_count(
            start_index, end_index)
        scaffold_ids_in_subtree: Set[np.int64] = set()
        if ct_exposed_segment.segment is not None:
            ct_exposed_segment.segment.reverse_subtree()

            def traverse_fn(n: ContigTree.Node):
                if n.contig_descriptor.scaffold_id is not None:
                    scaffold_ids_in_subtree.add(
                        n.contig_descriptor.scaffold_id)

            ContigTree.traverse_node(ct_exposed_segment.segment, traverse_fn)
        self.contig_tree.commit_exposed_segment(ct_exposed_segment)

        for scaffold_id in scaffold_ids_in_subtree:
            self.scaffold_holder.reverse_scaffold(scaffold_id)

        for resolution in self.resolutions:
            mt = self.matrix_trees[resolution]
            (start_bins, end_bins) = (
                borders_bins_start[resolution][0], borders_bins_end[resolution][1])
            mt.reverse_direction_in_bins(1 + start_bins, end_bins)
        self.clear_caches()

    def move_selection_range_bp(self, queried_start_contig_id: np.int64, queried_end_contig_id: np.int64,
                                target_start_order: np.int64) -> None:
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"

        queried_start_contig_order: np.int64 = self.contig_tree.get_contig_order(
            queried_start_contig_id)[1]
        queried_end_contig_order: np.int64 = self.contig_tree.get_contig_order(
            queried_end_contig_id)[1]

        if queried_end_contig_order < queried_start_contig_order:
            (queried_start_contig_id, queried_start_contig_order, queried_end_contig_id, queried_end_contig_order) = (
                queried_end_contig_id, queried_end_contig_order, queried_start_contig_id, queried_start_contig_order)

        queried_start_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_start_contig_id)
        queried_end_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_end_contig_id)

        queried_start_contig_scaffold_id = queried_start_node.contig_descriptor.scaffold_id
        queried_end_contig_scaffold_id = queried_end_node.contig_descriptor.scaffold_id

        start_contig_id: np.int64 = (
            queried_start_contig_id
            if queried_start_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(
                queried_start_contig_scaffold_id).scaffold_borders.start_contig_id
        )
        end_contig_id: np.int64 = (
            queried_end_contig_id
            if queried_end_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(queried_end_contig_scaffold_id).scaffold_borders.end_contig_id
        )

        (
            _,
            borders_bins_start,
            _,
            start_index
        ) = self.contig_tree.get_contig_location(start_contig_id)
        (
            _,
            borders_bins_end,
            _,
            end_index
        ) = self.contig_tree.get_contig_location(end_contig_id)

        if end_index < start_index:
            raise Exception(
                f"After selection was extended, its end contig with ID={end_contig_id} and order {end_index} precedes start contig with ID={start_contig_id} and order={start_index}")

        (previous_contig_scaffold,
         target_contig_scaffold,
         next_contig_scaffold,
         scaffold_starts_at_target,
         internal_scaffold_contig,
         scaffold_ends_at_target) = self.check_scaffold_borders_at_position(target_start_order)

        target_scaffold_id: Optional[np.int64] = None

        if internal_scaffold_contig:
            target_scaffold_id = target_contig_scaffold.scaffold_id

        ct_exposed_segment = self.contig_tree.expose_segment_by_count(
            start_index, end_index)
        if ct_exposed_segment.segment is not None:
            if target_scaffold_id is not None:
                ct_exposed_segment.segment.contig_descriptor.scaffold_id = target_scaffold_id
                ct_exposed_segment.segment.needs_updating_scaffold_id_in_subtree = True
        tmp_tree: Optional[ContigTree.Node] = self.contig_tree.merge_nodes(
            ct_exposed_segment.less, ct_exposed_segment.greater)
        l, r = self.contig_tree.split_node_by_count(
            tmp_tree, target_start_order)
        leftLength: Dict[np.int64, np.int64]
        if l is not None:
            leftLength = l.get_sizes()[0]
        else:
            leftLength = dict().fromkeys(self.resolutions, 0)
        self.contig_tree.commit_exposed_segment(
            ContigTree.ExposedSegment(l, ct_exposed_segment.segment, r))

        for resolution in self.resolutions:
            mt = self.matrix_trees[resolution]
            (start_bins, end_bins) = (
                borders_bins_start[resolution][0], borders_bins_end[resolution][1])
            mt.move_stripes(1+start_bins, end_bins, leftLength[resolution])
        self.clear_caches()

    def get_contig_location(self, contig_id: np.int64) -> Tuple[
        ContigDescriptor,
        Dict[np.int64, Tuple[np.int64, np.int64]],
        Dict[np.int64, Tuple[np.int64, np.int64]],
        np.int64
    ]:
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"
        return self.contig_tree.get_contig_location(contig_id)

    def process_scaffolds_during_move(
            self,
            target_order: np.int64,
            include_into_bordering_scaffold: bool,
    ):
        target_scaffold_id: Optional[np.int64] = None
        existing_contig_id: Optional[np.int64] = None
        # Check scaffold_id of existing contig currently with target_order:
        old_es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(
            target_order, target_order)

        # Check left border:
        if old_es.segment is not None:
            existing_contig_descriptor: ContigDescriptor = old_es.segment.contig_descriptor
            target_scaffold_id = existing_contig_descriptor.scaffold_id
            existing_contig_id = existing_contig_descriptor.contig_id
        self.contig_tree.commit_exposed_segment(old_es)

        # Check right border if requested to merge into scaffold:
        if target_scaffold_id is None and include_into_bordering_scaffold:
            old_previous_es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(
                target_order - 1,
                target_order - 1
            )
            if old_previous_es.segment is not None:
                existing_contig_descriptor: ContigDescriptor = old_previous_es.segment.contig_descriptor
                target_scaffold_id = existing_contig_descriptor.scaffold_id
                existing_contig_id = existing_contig_descriptor.contig_id
            self.contig_tree.commit_exposed_segment(old_es)

        moving_to_scaffold_border: bool = False
        moving_to_the_left_border: bool = False
        target_scaffold_descriptor: Optional[ScaffoldDescriptor] = None
        if target_scaffold_id is not None:
            target_scaffold_descriptor = self.scaffold_holder.get_scaffold_by_id(
                target_scaffold_id)
            assert existing_contig_id is not None, "Target scaffold id is determined without contig?"
            if existing_contig_id == target_scaffold_descriptor.scaffold_borders.start_contig_id:
                moving_to_the_left_border = True
                moving_to_scaffold_border = True
            elif existing_contig_id == target_scaffold_descriptor.scaffold_borders.end_contig_id:
                moving_to_scaffold_border = True
        return moving_to_scaffold_border, moving_to_the_left_border, target_scaffold_descriptor

    def check_scaffold_borders_at_position(
            self,
            target_order: np.int64,
    ) -> Tuple[
        Optional[ScaffoldDescriptor],
        Optional[ScaffoldDescriptor],
        Optional[ScaffoldDescriptor],
        bool,
        bool,
        bool
    ]:
        target_es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(
            target_order, target_order)
        previous_contig_node: Optional[ContigTree.Node] = ContigTree.get_rightmost(
            target_es.less)
        target_contig_node: Optional[ContigTree.Node] = target_es.segment
        next_contig_node: Optional[ContigTree.Node] = ContigTree.get_leftmost(
            target_es.greater)
        self.contig_tree.commit_exposed_segment(target_es)

        def get_node_scaffold_descriptor(node: Optional[ContigTree.Node]) -> Optional[ScaffoldDescriptor]:
            return self.get_scaffold_by_id(node.contig_descriptor.scaffold_id) if node is not None else None

        previous_contig_scaffold: Optional[ScaffoldDescriptor] = get_node_scaffold_descriptor(
            previous_contig_node)
        target_contig_scaffold: Optional[ScaffoldDescriptor] = get_node_scaffold_descriptor(
            target_contig_node)
        next_contig_scaffold: Optional[ScaffoldDescriptor] = get_node_scaffold_descriptor(
            next_contig_node)

        scaffold_starts_at_target: bool = (
            (target_contig_scaffold is not None)
            and
            (
                (previous_contig_scaffold is None)
                or
                (previous_contig_scaffold.scaffold_id !=
                 target_contig_scaffold.scaffold_id)
            )
        )

        internal_scaffold_contig: bool = (
            (
                (previous_contig_scaffold is not None)
                and
                (target_contig_scaffold is not None)
                and (next_contig_scaffold is not None)
            )
            and
            (
                previous_contig_scaffold.scaffold_id
                == target_contig_scaffold.scaffold_id
                == next_contig_scaffold.scaffold_id
            )
        )
        scaffold_ends_at_target: bool = (
            (target_contig_scaffold is not None)
            and
            (
                (next_contig_scaffold is None)
                or
                (target_contig_scaffold.scaffold_id !=
                 next_contig_scaffold.scaffold_id)
            )
        )

        return (
            previous_contig_scaffold,
            target_contig_scaffold,
            next_contig_scaffold,
            scaffold_starts_at_target,
            internal_scaffold_contig,
            scaffold_ends_at_target
        )

    def get_scaffold_by_id(self, scaffold_id: Optional[np.int64]) -> Optional[ScaffoldDescriptor]:
        return self.scaffold_housekeeping(scaffold_id) if scaffold_id is not None else None

    def scaffold_housekeeping(self, scaffold_id: np.int64) -> Optional[ScaffoldDescriptor]:
        try:
            scaffold_descriptor: ScaffoldDescriptor = self.scaffold_holder.get_scaffold_by_id(
                scaffold_id)
            scaffold_borders: Optional[ScaffoldBorders] = scaffold_descriptor.scaffold_borders
            if scaffold_borders is not None:
                start_contig_descriptor, _, _, _ = self.get_contig_location(
                    scaffold_borders.start_contig_id)
                end_contig_descriptor, _, _, _ = self.get_contig_location(
                    scaffold_borders.end_contig_id)
                start_scaffold_id: Optional[np.int64] = start_contig_descriptor.scaffold_id
                end_scaffold_id: Optional[np.int64] = end_contig_descriptor.scaffold_id
                assert (
                    (
                        start_scaffold_id == scaffold_id and end_scaffold_id == scaffold_id
                    ) or (
                        start_scaffold_id != scaffold_id and end_scaffold_id != scaffold_id
                    )
                ), f"Only one bordering contig belongs to the scaffold with id={scaffold_id}?"
                if (start_scaffold_id == scaffold_id) and (end_scaffold_id == scaffold_id):
                    return scaffold_descriptor
                elif (start_scaffold_id != scaffold_id) and (end_scaffold_id != scaffold_id):
                    self.scaffold_holder.remove_scaffold_by_id(scaffold_id)
                    return None
                else:
                    raise Exception(
                        f"Only one bordering contig belongs to the scaffold with id={scaffold_id}?")
        except KeyError:
            return None

    def group_contigs_into_scaffold(self, queried_start_contig_id: np.int64, queried_end_contig_id: np.int64,
                                    name: Optional[str] = None, spacer_length: int = 1000) -> ScaffoldDescriptor:
        queried_start_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_start_contig_id)
        queried_end_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_end_contig_id)

        queried_start_contig_scaffold_id = queried_start_node.contig_descriptor.scaffold_id
        queried_end_contig_scaffold_id = queried_end_node.contig_descriptor.scaffold_id

        start_contig_id: np.int64 = (
            queried_start_contig_id
            if queried_start_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(
                queried_start_contig_scaffold_id).scaffold_borders.start_contig_id
        )
        end_contig_id: np.int64 = (
            queried_end_contig_id
            if queried_end_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(queried_end_contig_scaffold_id).scaffold_borders.end_contig_id
        )

        new_scaffold: ScaffoldDescriptor = self.scaffold_holder.create_scaffold(
            name, ScaffoldDirection.FORWARD, spacer_length)
        new_scaffold.scaffold_borders = ScaffoldBorders(
            start_contig_id, end_contig_id)
        _, start_contig_order = self.contig_tree.get_contig_order(
            start_contig_id)
        _, end_contig_order = self.contig_tree.get_contig_order(
            end_contig_id)
        es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(
            start_contig_order, end_contig_order)
        if es.segment is not None:
            es.segment.contig_descriptor.scaffold_id = new_scaffold.scaffold_id
            es.segment.needs_updating_scaffold_id_in_subtree = True
            self.scaffold_holder.remove_scaffold_by_id(
                queried_start_contig_scaffold_id)
            self.scaffold_holder.remove_scaffold_by_id(
                queried_end_contig_scaffold_id)
        self.contig_tree.commit_exposed_segment(es)
        self.clear_caches()
        return new_scaffold

    def ungroup_contigs_from_scaffold(self, queried_start_contig_id: np.int64, queried_end_contig_id: np.int64) -> None:
        queried_start_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_start_contig_id)
        queried_end_node: ContigTree.Node = self.contig_tree.get_updated_contig_node_by_contig_id(
            queried_end_contig_id)

        queried_start_contig_scaffold_id = queried_start_node.contig_descriptor.scaffold_id
        queried_end_contig_scaffold_id = queried_end_node.contig_descriptor.scaffold_id

        start_contig_id: np.int64 = (
            queried_start_contig_id
            if queried_start_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(
                queried_start_contig_scaffold_id).scaffold_borders.start_contig_id
        )
        end_contig_id: np.int64 = (
            queried_end_contig_id
            if queried_end_contig_scaffold_id is None
            else self.scaffold_holder.get_scaffold_by_id(queried_end_contig_scaffold_id).scaffold_borders.end_contig_id
        )

        _, start_contig_order = self.contig_tree.get_contig_order(
            start_contig_id)
        _, end_contig_order = self.contig_tree.get_contig_order(
            end_contig_id)
        es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(
            start_contig_order, end_contig_order)
        if es.segment is not None:
            es.segment.contig_descriptor.scaffold_id = None
            es.segment.needs_updating_scaffold_id_in_subtree = True
            self.scaffold_holder.remove_scaffold_by_id(
                queried_start_contig_scaffold_id)
            self.scaffold_holder.remove_scaffold_by_id(
                queried_end_contig_scaffold_id)
        self.contig_tree.commit_exposed_segment(es)
        self.clear_caches()

    def dump_stripe_info(self, f: h5py.File):
        for resolution in self.resolutions:
            # TODO: Rename 'contigs' group in something like 'metadata'
            stripe_info_group: h5py.Group = f[f'/resolutions/{resolution}/contigs']
            stripe_count: np.int64 = self.matrix_trees[resolution].get_node_count(
            )

            ordered_stripe_ids: h5py.Dataset = create_dataset_if_not_exists(
                'ordered_stripe_ids', stripe_info_group, shape=(stripe_count,),
                maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args
            )

            # stripe_direction: h5py.Dataset = create_dataset_if_not_exists(
            #     'stripe_direction', stripe_info_group, shape=(stripe_count,),
            #     maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
            # )

            # stripe_hide_type: h5py.Dataset = create_dataset_if_not_exists(
            #     'stripe_hide_type', stripe_info_group, shape=(stripe_count,),
            #     maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
            # )

            ordered_stripe_ids_backup: h5py.Dataset = create_dataset_if_not_exists(
                'ordered_stripe_ids_backup', stripe_info_group, shape=(stripe_count,),
                maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args
            )

            # stripe_direction_backup: h5py.Dataset = create_dataset_if_not_exists(
            #     'stripe_direction_backup', stripe_info_group, shape=(stripe_count,),
            #     maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
            # )

            # stripe_hide_type_backup: h5py.Dataset = create_dataset_if_not_exists(
            #     'stripe_hide_type_backup', stripe_info_group, shape=(stripe_count,),
            #     maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
            # )

            get_attribute_value_or_create_if_not_exists(
                'stripe_backup_done', False, stripe_info_group)

            if get_attribute_value_or_create_if_not_exists('stripe_write_finished', False, stripe_info_group):
                stripe_info_group.attrs['stripe_backup_done'] = False
                stripe_info_group.attrs['stripe_write_finished'] = False

            stripe_write_finished: bool = stripe_info_group.attrs.get(
                'stripe_write_finished')
            assert not stripe_write_finished, "Incorrect state of writing changes?"

            stripe_backup_done: bool = stripe_info_group.attrs.get(
                'stripe_backup_done')

            if not stripe_backup_done:
                ordered_stripe_ids_backup[:] = ordered_stripe_ids[:]
                stripe_info_group.attrs['stripe_backup_done'] = True

            ordered_stripe_ids_list: List[np.int64] = []
            stripe_direction_list: np.ndarray = np.ones(
                shape=(stripe_count,), dtype=np.int8)

            # stripe_hide_type_list: np.ndarray = np.zeros(shape=(stripe_count,), dtype=np.int8)

            def traversal_fn(node: StripeTree.Node) -> None:
                stripe_descriptor: StripeDescriptor = node.stripe_descriptor
                stripe_id: np.int64 = stripe_descriptor.stripe_id
                ordered_stripe_ids_list.append(stripe_id)

            self.matrix_trees[resolution].traverse(traversal_fn)

            ordered_stripe_ids[:] = ordered_stripe_ids_list[:]
            # stripe_direction[:] = stripe_direction_list[:]

            stripe_info_group.attrs['stripe_write_finished'] = True

    def dump_contig_info(
            self,
            f: h5py.File
    ) -> np.ndarray:
        contig_count: np.int64 = self.contig_tree.get_node_count()
        contig_info_group: h5py.Group = f['/contig_info']
        ordered_contig_ids: h5py.Dataset = create_dataset_if_not_exists(
            'ordered_contig_ids', contig_info_group, shape=(contig_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args
        )

        contig_direction: h5py.Dataset = create_dataset_if_not_exists(
            'contig_direction', contig_info_group, shape=(contig_count,),
            maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
        )

        contig_scaffold_id: h5py.Dataset = create_dataset_if_not_exists(
            'contig_scaffold_id', contig_info_group, shape=(contig_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args
        )

        ordered_contig_ids_backup: h5py.Dataset = create_dataset_if_not_exists(
            'ordered_contig_ids_backup', contig_info_group, shape=(contig_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args
        )

        contig_direction_backup: h5py.Dataset = create_dataset_if_not_exists(
            'contig_direction_backup', contig_info_group, shape=(contig_count,),
            maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
        )

        contig_scaffold_id_backup: h5py.Dataset = create_dataset_if_not_exists(
            'contig_scaffold_id_backup', contig_info_group, shape=(contig_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args
        )

        resolution_to_contig_hide_type: Dict[np.int64, h5py.Dataset] = dict()
        resolution_to_contig_hide_type_backup: Dict[np.int64, h5py.Dataset] = dict(
        )

        for resolution in self.resolutions:
            contig_hide_type: h5py.Dataset = create_dataset_if_not_exists(
                'contig_hide_type', f[f'/resolutions/{resolution}/contigs'], shape=(contig_count,),
                maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
            )
            contig_hide_type_backup: h5py.Dataset = create_dataset_if_not_exists(
                'contig_hide_type_backup', f[f'/resolutions/{resolution}/contigs'], shape=(contig_count,),
                maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
            )
            resolution_to_contig_hide_type[resolution] = contig_hide_type
            resolution_to_contig_hide_type_backup[resolution] = contig_hide_type_backup

        get_attribute_value_or_create_if_not_exists(
            'contig_backup_done', False, contig_info_group)

        if get_attribute_value_or_create_if_not_exists('contig_write_finished', False, contig_info_group):
            contig_info_group.attrs['contig_backup_done'] = False
            contig_info_group.attrs['contig_write_finished'] = False

        contig_write_finished: bool = contig_info_group.attrs.get(
            'contig_write_finished')
        assert not contig_write_finished, "Incorrect state of writing changes?"

        contig_backup_done: bool = contig_info_group.attrs.get(
            'contig_backup_done')

        if not contig_backup_done:
            ordered_contig_ids_backup[:] = ordered_contig_ids[:]
            contig_direction_backup[:] = contig_direction[:]
            contig_scaffold_id_backup[:] = contig_scaffold_id[:]
            for resolution in self.resolutions:
                resolution_to_contig_hide_type[resolution][:
                                                           ] = resolution_to_contig_hide_type_backup[resolution][:]
            contig_info_group.attrs['contig_backup_done'] = True

        ordered_contig_ids_list: List[np.int64] = []
        contig_direction_list: np.ndarray = np.ones(
            shape=(contig_count,), dtype=np.int8)
        # ContigId -> [Resolution -> ContigHideType]
        contig_hide_types: List[Dict[np.int64, ContigHideType]] = [
            None] * contig_count
        contig_old_scaffold_id: np.ndarray = np.zeros(
            shape=(contig_count,), dtype=np.int64)
        used_scaffold_ids: Set[np.int64] = set()

        def traversal_fn(node: ContigTree.Node) -> None:
            contig_descriptor: ContigDescriptor = node.contig_descriptor
            contig_id: np.int64 = contig_descriptor.contig_id
            ordered_contig_ids_list.append(contig_id)
            s_id = contig_descriptor.scaffold_id
            if s_id is None:
                contig_old_scaffold_id[contig_id] = -1
            else:
                contig_old_scaffold_id[contig_id] = s_id
                used_scaffold_ids.add(s_id)
            contig_direction_list[contig_id] = contig_descriptor.direction.value
            contig_hide_types[contig_id] = contig_descriptor.presence_in_resolution

        self.contig_tree.traverse(traversal_fn)

        self.scaffold_holder.remove_unused_scaffolds(used_scaffold_ids)
        scaffold_old_id_to_new_id: Dict[np.int64, np.int64] = dict()
        scaffold_new_id_to_old_id: np.ndarray = np.zeros(
            shape=(len(used_scaffold_ids),), dtype=np.int64)
        for new_id, old_id in enumerate(used_scaffold_ids):
            scaffold_old_id_to_new_id[old_id] = new_id
            scaffold_new_id_to_old_id[new_id] = old_id

        ordered_contig_ids[:] = ordered_contig_ids_list[:]
        contig_direction[:] = contig_direction_list[:]
        contig_scaffold_id[:] = list(map(lambda old_s_id: scaffold_old_id_to_new_id[old_s_id] if old_s_id != -1 else -1,
                                         contig_old_scaffold_id))[:]

        for resolution in self.resolutions:
            contig_id_to_contig_hide_type_at_resolution = np.ones(
                shape=(contig_count,), dtype=np.int8)
            for contig_id, contig_presence_in_resolution in enumerate(contig_hide_types):
                contig_id_to_contig_hide_type_at_resolution[
                    contig_id] = contig_presence_in_resolution[resolution].value
            resolution_to_contig_hide_type[resolution][:
                                                       ] = contig_id_to_contig_hide_type_at_resolution[:]

        contig_info_group.attrs['contig_write_finished'] = True

        return scaffold_new_id_to_old_id

    def dump_scaffold_info(
            self,
            f: h5py.File,
            scaffold_new_id_to_old_id: np.ndarray,
    ) -> None:
        scaffold_info_group: h5py.Group = create_group_if_not_exists(
            'scaffold_info', f['/'])
        scaffold_count: np.int64 = len(self.scaffold_holder.scaffold_table)

        scaffold_name_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_name', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=h5py.string_dtype(encoding='utf8'), **additional_dataset_creation_args,
        )

        scaffold_start_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_start', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args,
        )

        scaffold_end_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_end', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args,
        )

        scaffold_direction_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_direction', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args,
        )

        scaffold_spacer_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_spacer', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args,
        )

        scaffold_name_backup_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_name_backup', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=h5py.string_dtype(encoding='utf8'), **additional_dataset_creation_args,
        )

        scaffold_start_backup_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_start_backup', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args,
        )

        scaffold_end_backup_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_end_backup', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args,
        )

        scaffold_direction_backup_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_direction_backup', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args,
        )

        scaffold_spacer_backup_ds: h5py.Dataset = create_dataset_if_not_exists(
            'scaffold_spacer_backup', scaffold_info_group, shape=(scaffold_count,),
            maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args,
        )

        get_attribute_value_or_create_if_not_exists(
            'scaffold_backup_done', False, scaffold_info_group)

        if get_attribute_value_or_create_if_not_exists('scaffold_write_finished', False, scaffold_info_group):
            scaffold_info_group.attrs['scaffold_backup_done'] = False
            scaffold_info_group.attrs['scaffold_write_finished'] = False

        scaffold_write_finished: bool = scaffold_info_group.attrs.get(
            'scaffold_write_finished')
        assert not scaffold_write_finished, "Incorrect state of writing changes?"

        scaffold_backup_done: bool = scaffold_info_group.attrs.get(
            'scaffold_backup_done')

        if not scaffold_backup_done:
            scaffold_name_backup_ds[:] = scaffold_name_ds[:]
            scaffold_start_backup_ds[:] = scaffold_start_ds[:]
            scaffold_end_backup_ds[:] = scaffold_end_ds[:]
            scaffold_direction_backup_ds[:] = scaffold_direction_ds[:]
            scaffold_spacer_backup_ds[:] = scaffold_spacer_ds[:]
            scaffold_info_group.attrs['scaffold_backup_done'] = True

        scaffold_names: List[str] = []
        scaffold_starts: np.ndarray = np.zeros(
            shape=(scaffold_count,), dtype=np.int64)
        scaffold_ends: np.ndarray = np.zeros(
            shape=(scaffold_count,), dtype=np.int64)
        scaffold_directions: np.ndarray = np.zeros(
            shape=(scaffold_count,), dtype=np.int8)
        scaffold_spacers: np.ndarray = np.zeros(
            shape=(scaffold_count,), dtype=np.int32)

        for new_id, old_id in enumerate(scaffold_new_id_to_old_id):
            scaffold_descriptor: ScaffoldDescriptor = self.scaffold_holder.get_scaffold_by_id(
                old_id)
            scaffold_names.append(scaffold_descriptor.scaffold_name)
            if scaffold_descriptor.scaffold_borders is not None:
                scaffold_starts[new_id] = scaffold_descriptor.scaffold_borders.start_contig_id
                scaffold_ends[new_id] = scaffold_descriptor.scaffold_borders.end_contig_id
            else:
                scaffold_starts[new_id], scaffold_ends[new_id] = -1, -1
            scaffold_directions[new_id] = scaffold_descriptor.scaffold_direction.value
            scaffold_spacers[new_id] = scaffold_descriptor.spacer_length

        scaffold_name_ds.resize(scaffold_count, 0)
        scaffold_name_ds[:] = scaffold_names[:]

        scaffold_start_ds.resize(scaffold_count, 0)
        scaffold_start_ds[:] = scaffold_starts[:]

        scaffold_end_ds.resize(scaffold_count, 0)
        scaffold_end_ds[:] = scaffold_ends[:]

        scaffold_direction_ds.resize(scaffold_count, 0)
        scaffold_direction_ds[:] = scaffold_directions[:]

        scaffold_spacer_ds.resize(scaffold_count, 0)
        scaffold_spacer_ds[:] = scaffold_spacers[:]

        scaffold_info_group.attrs['scaffold_write_finished'] = True

    def save(self) -> None:
        with self.hdf_file_lock.gen_wlock():
            try:
                self.opened_hdf_file.close()
                self.opened_hdf_file = h5py.File(self.filepath, mode='a')
                f = self.opened_hdf_file
                self.dump_stripe_info(f)
                f.flush()
                scaffold_new_id_to_old_id = self.dump_contig_info(f)
                f.flush()
                self.dump_scaffold_info(f, scaffold_new_id_to_old_id)
                f.flush()
                self.clear_caches(saved_blocks=True)
                self.opened_hdf_file.close()
            except Exception as e:
                print(
                    f"Exception was thrown during save process: {str(e)}\nFile might be saved incorrectly.")
                self.state = ChunkedFile.FileState.INCORRECT
                raise e
            finally:
                self.opened_hdf_file = h5py.File(self.filepath, mode='r')

    def close(self, need_save: bool = True) -> None:
        if need_save:
            self.save()
        self.state = ChunkedFile.FileState.CLOSED

    def link_fasta(self, fasta_filename: str) -> None:
        with self.fasta_file_lock.gen_wlock():
            if self.fasta_processor is not None:
                print("Warning: re-linking FASTA file")
                del self.fasta_processor
            self.fasta_processor = FASTAProcessor(fasta_filename)

    def get_fasta_for_assembly(self, writable_stream) -> None:
        with self.fasta_file_lock.gen_rlock():
            if self.fasta_processor is None:
                raise Exception("FASTA File is not linked")

            ordered_contig_descriptors: List[ContigDescriptor] = []

            def traverse_fn(node: ContigTree.Node) -> None:
                ordered_contig_descriptors.append(node.contig_descriptor)

            self.contig_tree.traverse(traverse_fn)

            ordered_finalization_records: List[Tuple[FinalizeRecordType, List[ContigDescriptor]]] = [
            ]

            for contig_descriptor in ordered_contig_descriptors:
                if contig_descriptor.scaffold_id is None:
                    ordered_finalization_records.append((
                        FinalizeRecordType.CONTIG_NOT_IN_SCAFFOLD,
                        [contig_descriptor]
                    ))
                else:
                    last_scaffold_id: Optional[np.int64] = None
                    if len(ordered_finalization_records) > 0:
                        last_scaffold_id = ordered_finalization_records[-1][1][-1].scaffold_id
                    if contig_descriptor.scaffold_id is not None and contig_descriptor.scaffold_id == last_scaffold_id:
                        assert (
                            ordered_finalization_records[-1][0] == FinalizeRecordType.SCAFFOLD
                        ), "Last contig descriptor has a scaffold_id but marked as out-of-scaffold?"
                        ordered_finalization_records[-1][1].append(
                            contig_descriptor)
                    else:
                        ordered_finalization_records.append((
                            FinalizeRecordType.SCAFFOLD,
                            [contig_descriptor]
                        ))

            self.fasta_processor.finalize_fasta_for_assembly(
                writable_stream,
                ordered_finalization_records,
                self.scaffold_holder.scaffold_table,
                self.contig_names
            )

    def load_assembly_from_agp(self, agp_filepath: Path) -> None:
        agpParser: AGPparser = AGPparser(agp_filepath.absolute())
        contig_records = agpParser.getAGPContigRecords()
        scaffold_records = agpParser.getAGPScaffoldRecords()

        requestedIdToContigDirection: Dict[np.int64, ContigDirection] = dict()

        for contig_record in contig_records:
            contig_id = self.contig_name_to_contig_id[contig_record.name]
            requestedIdToContigDirection[contig_id] = contig_record.direction

        contigIdsToBeRotated: List[np.int64] = []

        def traverse_directions(n: ContigTree.Node) -> None:
            if n.contig_descriptor.direction != requestedIdToContigDirection[n.contig_descriptor.contig_id]:
                contigIdsToBeRotated.append(n.contig_descriptor.contig_id)

        # Clear previous scaffolds info to rotate segments
        if self.contig_tree.root is not None:
            self.contig_tree.root.contig_descriptor.scaffold_id = None
            self.contig_tree.root.needs_updating_scaffold_id_in_subtree = True

        self.scaffold_holder.clear()

        self.contig_tree.traverse(traverse_directions)

        for contig_order, contig_record in enumerate(contig_records):
            contig_id = self.contig_name_to_contig_id[contig_record.name]
            try:
                self.move_selection_range(contig_id, contig_id, contig_order)
            except AssertionError as e:
                # raise e
                self.move_selection_range(contig_id, contig_id, contig_order)

        for contig_id in contigIdsToBeRotated:
            self.reverse_selection_range(contig_id, contig_id)

        for scaffold_record in scaffold_records:
            start_contig_id: np.int64 = self.contig_name_to_contig_id[scaffold_record.start_ctg]
            end_contig_id: np.int64 = self.contig_name_to_contig_id[scaffold_record.end_ctg]
            sd: ScaffoldDescriptor = self.scaffold_holder.create_scaffold(
                scaffold_record.name, scaffold_borders=ScaffoldBorders(
                    start_contig_id,
                    end_contig_id
                )
            )
            _, start_order = self.contig_tree.get_contig_order(start_contig_id)
            _, end_order = self.contig_tree.get_contig_order(end_contig_id)
            es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(
                start_order, end_order)
            es.segment.contig_descriptor.scaffold_id = sd.scaffold_id
            es.segment.needs_updating_scaffold_id_in_subtree = True
            self.contig_tree.commit_exposed_segment(es)

    def get_agp_for_assembly(self, writable_stream) -> None:
        agp_export_processor: AGPExporter = AGPExporter()

        ordered_contig_descriptors: List[ContigDescriptor] = []

        def traverse_fn(node: ContigTree.Node) -> None:
            ordered_contig_descriptors.append(node.contig_descriptor)

        self.contig_tree.traverse(traverse_fn)

        agp_export_processor.exportAGP(
            writable_stream,
            ordered_contig_descriptors,
            self.scaffold_holder.scaffold_table
        )

    def get_contig_descriptors_in_range(
            self,
            from_bp_incl: np.int64,
            to_bp_incl: np.int64
    ) -> Tuple[
        List[ContigDescriptor],
        np.int64,
        np.int64,
    ]:
        es: ContigTree.ExposedSegment = self.contig_tree.expose_segment(resolution=np.int64(
            0), start=from_bp_incl, end=to_bp_incl, units=QueryLengthUnit.BASE_PAIRS)
        descriptors: List[ContigDescriptor] = []

        def traverse_fn(n: ContigTree.Node) -> None:
            descriptors.append(n.contig_descriptor)
            pass

        self.contig_tree.commit_exposed_segment(es)

        self.contig_tree.traverse_node(es.segment, traverse_fn)

        _, start_contig_location_in_resolutions, _, _ = self.get_contig_location(
            descriptors[0].contig_id)
        _, end_contig_location_in_resolutions, _, _ = self.get_contig_location(
            descriptors[-1].contig_id)

        start_offset_bp: np.int64 = from_bp_incl - \
            start_contig_location_in_resolutions[0][0]
        end_offset_bp: np.int64 = end_contig_location_in_resolutions[0][1] - to_bp_incl

        return (
            descriptors,
            start_offset_bp,
            end_offset_bp
        )

    def get_fasta_for_selection(
            self, from_bp_x_incl: np.int64, to_bp_x_incl: np.int64,
            from_bp_y_incl: np.int64, to_bp_y_incl: np.int64,
            buf: BytesIO
    ) -> None:
        (
            descriptorsX,
            start_offset_bpX,
            end_offset_bpX,
        ) = self.get_contig_descriptors_in_range(from_bp_incl=from_bp_x_incl, to_bp_incl=to_bp_x_incl)
        (
            descriptorsY,
            start_offset_bpY,
            end_offset_bpY,
        ) = self.get_contig_descriptors_in_range(from_bp_incl=from_bp_y_incl, to_bp_incl=to_bp_y_incl)

        self.fasta_processor.get_fasta_for_range(
            buf, descriptorsX, f"{from_bp_x_incl}bp-{to_bp_x_incl}bp", start_offset_bpX, end_offset_bpX)
        self.fasta_processor.get_fasta_for_range(
            buf, descriptorsY, f"{from_bp_y_incl}bp-{to_bp_y_incl}bp", start_offset_bpY, end_offset_bpY)
