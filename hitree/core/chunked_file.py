import itertools
import threading
from enum import Enum
from threading import Lock
from typing import Optional, Dict, Tuple, List, Set

import h5py
import numpy as np
from cachetools import LRUCache, cachedmethod
from cachetools.keys import hashkey
from readerwriterlock import rwlock
from scipy.sparse import coo_matrix

from hitree.core.FASTAProcessor import FASTAProcessor
from hitree.core.common import ContigDirection, StripeDescriptor, ContigDescriptor, ScaffoldDescriptor, ScaffoldBorders, \
    ScaffoldDirection, FinalizeRecordType, StripeDirection, ContigHideType
from hitree.core.contig_tree import ContigTree
from hitree.core.scaffold_holder import ScaffoldHolder
from hitree.core.stripe_tree import StripeTree
from hitree.util.h5helpers import create_dataset_if_not_exists, get_attribute_value_or_create_if_not_exists, \
    create_group_if_not_exists

additional_dataset_creation_args = {
    'compression': 'lzf',
    'shuffle': True,
    'chunks': True,
}


def constrain_coordinate(x_bins: np.int64, lower: np.int64, upper: np.int64) -> np.int64:
    return max(min(x_bins, upper), lower)


class ChunkedFile(object):
    class FileState(Enum):
        CLOSED = 0
        OPENED = 1
        INCORRECT = 2

    def __init__(
            self,
            filepath: str,
            block_cache_size: int = 256
    ) -> None:
        super().__init__()
        self.filepath: str = filepath
        self.matrix_trees: Dict[np.int64, StripeTree] = dict()
        self.contig_names: List[str] = []
        self.contig_name_to_contig_id: Dict[str, np.int64] = dict()
        self.contig_lengths_bp: Dict[np.int64, np.int64] = dict()
        self.resolutions: List[np.int64] = []
        self.contig_tree: Optional[ContigTree] = None
        self.state: ChunkedFile.FileState = ChunkedFile.FileState.CLOSED
        self.dense_submatrix_size: Dict[np.int64, np.int64] = dict()  # Resolution -> MSS
        self.block_cache_size = block_cache_size
        self.block_cache = LRUCache(maxsize=self.block_cache_size)
        self.block_intersection_cache = LRUCache(maxsize=self.block_cache_size)
        self.block_cache_lock: Lock = threading.Lock()
        self.block_intersection_cache_lock: Lock = threading.Lock()
        self.scaffold_holder: ScaffoldHolder = ScaffoldHolder()
        self.dtype: Optional[np.dtype] = None
        self.hdf_file_lock: rwlock.RWLockWrite = rwlock.RWLockWrite()
        self.fasta_processor: Optional[FASTAProcessor] = None
        self.fasta_file_lock: rwlock.RWLockFair = rwlock.RWLockFair()

    def open(self):
        # NOTE: When file is opened in this method, we assert no one writes to it
        contig_id_to_length_by_resolution: Dict[np.int64, Dict[np.int64, np.int64]] = dict()
        contig_id_to_hide_type_by_resolution: Dict[np.int64, Dict[np.int64, ContigHideType]] = dict()
        contig_id_to_direction: List[ContigDirection] = []
        contig_id_to_scaffold_id: List[Optional[np.int64]] = []
        ordered_contig_ids: np.ndarray
        with self.hdf_file_lock.gen_rlock(), h5py.File(self.filepath, mode='r') as f:
            resolutions = np.array(
                [np.int64(sdn) for sdn in sorted(filter(lambda s: s.isnumeric(), f['resolutions'].keys()))],
                dtype=np.int64
            )
            self.resolutions = resolutions
            self.dtype = f[f'resolutions/{min(resolutions)}/treap_coo/block_vals'].dtype

            (
                contig_id_to_contig_length_bp,
                resolution_to_contig_length_bins,
                resolution_to_contig_hide_type,
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
                contig_id_to_contig_length_bins_at_resolution = resolution_to_contig_length_bins[resolution]
                contig_id_to_contig_hide_type_at_resolution = resolution_to_contig_hide_type[resolution]
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

            contig_info_group: h5py.Group = f[f'/contig_info/']
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
                contig_id_to_direction.append(ContigDirection(contig_direction))
                contig_id_to_scaffold_id.append(contig_scaff_id if contig_scaff_id >= 0 else None)

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

                for res in resolutions[1:]:
                    if contig_id_to_contig_length_bp[contig_id] < res:
                        contig_presence_at_resolution[res] = ContigHideType(max(
                            contig_presence_at_resolution[res].value,
                            ContigHideType.AUTO_HIDDEN.value
                        ))
                contig_descriptor: ContigDescriptor = ContigDescriptor.make_contig_descriptor(
                    contig_id=contig_id,
                    direction=contig_id_to_direction[contig_id],
                    contig_length_bp=contig_id_to_contig_length_bp[contig_id],
                    contig_length_at_resolution=resolution_to_contig_length,
                    contig_presence_in_resolution=contig_presence_at_resolution,
                    scaffold_id=contig_id_to_scaffold_id[contig_id]
                )
                contig_id_to_contig_descriptor.append(contig_descriptor)

            for contig_id in ordered_contig_ids:
                contig_descriptor = contig_id_to_contig_descriptor[contig_id]
                self.contig_tree.insert_at_position(contig_descriptor, self.contig_tree.get_node_count())

            for resolution in resolutions:
                # Construct matrix tree:
                (
                    self.matrix_trees[resolution],
                    self.dense_submatrix_size[resolution]
                ) = self.read_stripe_data(f, resolution)

            self.restore_scaffolds(f)

        self.state = ChunkedFile.FileState.OPENED

    def clear_caches(self, saved_blocks: bool = False):
        if saved_blocks:
            if self.load_saved_dense_block.cache_lock is not None:
                with self.load_saved_dense_block.cache_lock(self):
                    self.load_saved_dense_block.cache(self).clear()
        if self.get_block_intersection_as_dense_matrix.cache_lock is not None:
            with self.get_block_intersection_as_dense_matrix.cache_lock(self):
                self.get_block_intersection_as_dense_matrix.cache(self).clear()

    def restore_scaffolds(self, f: h5py.File):
        if 'scaffold_info' not in f['/'].keys():
            # No scaffolds are present
            return
        scaffold_info_group: h5py.Group = f['/scaffold_info']
        scaffold_name_ds: h5py.Dataset = scaffold_info_group['scaffold_name']

        scaffold_start_ds: h5py.Dataset = scaffold_info_group['scaffold_name']
        scaffold_end_ds: h5py.Dataset = scaffold_info_group['scaffold_name']

        scaffold_direction_ds: h5py.Dataset = scaffold_info_group['scaffold_name']
        scaffold_spacer_ds: h5py.Dataset = scaffold_info_group['scaffold_name']

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
                (scaffold_end_contig_id == -1) if (scaffold_start_contig_id == -1) else (scaffold_end_contig_id != -1)
            ), "Scaffold borders are existent/nonexistent separately??"
            scaffold_descriptor: ScaffoldDescriptor = ScaffoldDescriptor(
                scaffold_id=scaffold_id,
                scaffold_name=scaffold_name,
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

    def read_contig_data(
            self,
            f: h5py.File
    ) -> Tuple[
        np.ndarray,
        Dict[np.int64, np.ndarray],
        Dict[np.int64, np.ndarray],
        List[str]
    ]:
        contig_info_group: h5py.Group = f[f'/contig_info/']
        contig_names_ds: h5py.Dataset = contig_info_group['contig_name']
        contig_lengths_bp: h5py.Dataset = contig_info_group['contig_length_bp']

        contig_count: np.int64 = len(contig_names_ds)

        assert len(contig_lengths_bp) == contig_count, "Different contig count in different datasets??"

        # Resolution -> [ContigId -> ContigLengthBp]
        resolution_to_contig_length_bins: Dict[np.int64, np.ndarray] = dict()
        # Resolution -> [ContigId -> ContigHideType]
        resolution_to_contig_hide_type: Dict[np.int64, np.ndarray] = dict()
        for resolution in self.resolutions:
            resolution_group: h5py.Group = f[f'/resolutions/{resolution}/']
            contig_length_bins_ds: h5py.Dataset = resolution_group['contigs/contig_length_bins']
            contig_hide_type_ds: h5py.Dataset = resolution_group['contigs/contig_hide_type']
            assert len(contig_length_bins_ds) == contig_count, "Different contig count in different datasets??"

            resolution_to_contig_length_bins[resolution] = np.array(
                contig_length_bins_ds[:].astype(np.int64),
                dtype=np.int64
            )

            resolution_to_contig_hide_type[resolution] = np.array(
                contig_hide_type_ds[:].astype(np.int64),
                dtype=np.int8
            )

        contig_id_to_contig_length_bp: np.ndarray = np.array(contig_lengths_bp[:].astype(np.int64), dtype=np.int64)
        contig_names: List[str] = [bytes(contig_name).decode('utf-8') for contig_name in contig_names_ds]

        return (
            contig_id_to_contig_length_bp,
            resolution_to_contig_length_bins,
            resolution_to_contig_hide_type,
            contig_names
        )

    def read_stripe_data(
            self,
            f: h5py.File,
            resolution: np.int64
    ) -> Tuple[
        StripeTree,
        np.int64
    ]:
        resolution_group: h5py.Group = f[f'/resolutions/{resolution}/']
        treap_coo_group: h5py.Group = resolution_group['treap_coo']
        dense_submatrix_size: np.int64 = treap_coo_group.attrs.get('dense_submatrix_size')
        stripes_group: h5py.Group = f[f'/resolutions/{resolution}/stripes']
        stripe_directions: h5py.Dataset = stripes_group['stripe_direction']
        ordered_stripe_ids_ds: h5py.Dataset = stripes_group['ordered_stripe_ids']
        stripe_lengths_bins: h5py.Dataset = stripes_group['stripe_length_bins']
        stripe_lengths_bp: h5py.Dataset = stripes_group['stripe_length_bp']
        stripe_id_to_contig_id: h5py.Dataset = stripes_group['stripes_contig_id']

        stripe_tree = StripeTree()

        stripe_descriptors: List[StripeDescriptor] = [
            StripeDescriptor.make_stripe_descriptor(
                stripe_id,
                stripe_length_bins,
                stripe_length_bp,
                StripeDirection(direction),
                stripes_contig_id
            ) for stripe_id, (
                stripe_length_bins,
                stripe_length_bp,
                direction,
                stripes_contig_id
            ) in enumerate(
                zip(
                    stripe_lengths_bins,
                    stripe_lengths_bp,
                    stripe_directions,
                    stripe_id_to_contig_id
                )
            )
        ]

        for stripe_id in ordered_stripe_ids_ds:
            stripe_tree.insert_at_position(stripe_tree.get_node_count(), stripe_descriptors[stripe_id])
        return stripe_tree, dense_submatrix_size

    def get_stripes_for_range(
            self,
            resolution: np.int64,
            start_bins: np.int64,
            end_bins: np.int64
    ) -> Tuple[
        np.int64,
        List[StripeDescriptor]
    ]:
        tree = self.matrix_trees[resolution]
        exposed_segment: StripeTree.ExposedSegment = tree.expose_segment(start_bins, end_bins)
        t: Tuple[
            Optional[StripeTree.Node],
            Optional[StripeTree.Node],
            Optional[StripeTree.Node]
        ] = (exposed_segment.less, exposed_segment.segment, exposed_segment.greater)
        (blocks_less, segment_blocks, _) = t
        if segment_blocks is None:
            return 0, []
        zero_at_segment_bins: np.int64 = blocks_less.get_sizes().length_bins if blocks_less is not None else 0
        blocks: List[StripeDescriptor] = []

        def traverse_fn(node: StripeTree.Node) -> None:
            blocks.append(node.stripe_descriptor)

        StripeTree.traverse_node(segment_blocks, traverse_fn)
        tree.commit_exposed_segment(exposed_segment)
        return zero_at_segment_bins, blocks

    @cachedmethod(cache=lambda s: s.block_cache,
                  key=lambda s, f, r, rb, cb: hashkey((r, rb.stripe_id, cb.stripe_id)),
                  lock=lambda s: s.block_cache_lock)
    def load_saved_dense_block(
            self,
            f: h5py.File,
            resolution: np.int64,
            row_block: StripeDescriptor,
            col_block: StripeDescriptor
    ) -> Tuple[np.ndarray, bool]:
        r: np.int64 = row_block.stripe_id
        c: np.int64 = col_block.stripe_id

        blocks_dir: h5py.Group = f[f'/resolutions/{resolution}/treap_coo']
        stripes_count: np.int64 = blocks_dir.attrs['stripes_count']
        block_index_in_datasets: np.int64 = r * stripes_count + c

        block_lengths: h5py.Dataset = blocks_dir['block_length']
        block_length = block_lengths[block_index_in_datasets]
        is_empty: bool = (block_length <= 0)

        result_matrix: np.ndarray

        block_vals: h5py.Dataset = blocks_dir['block_vals']

        if is_empty:
            # This block was empty and not exported, so it is all zeros
            result_matrix = np.zeros(shape=(row_block.stripe_length_bins, col_block.stripe_length_bins),
                                     dtype=block_vals.dtype)
        else:
            block_offsets: h5py.Dataset = blocks_dir['block_offset']
            block_offset = block_offsets[block_index_in_datasets]

            is_dense: bool = (block_offset < 0)

            if is_dense:
                dense_blocks: h5py.Dataset = blocks_dir['dense_blocks']
                index_in_dense_blocks: np.int64 = -(block_offset + 1)
                result_matrix = dense_blocks[index_in_dense_blocks, 0, :, :]
                pass
            else:
                block_finish = block_offset + block_length
                block_rows: h5py.Dataset = blocks_dir['block_rows']
                block_cols: h5py.Dataset = blocks_dir['block_cols']
                mx = coo_matrix(
                    (
                        block_vals[block_offset:block_finish],
                        (
                            block_rows[block_offset:block_finish],
                            block_cols[block_offset:block_finish]
                        )
                    ),
                    shape=(row_block.stripe_length_bins, col_block.stripe_length_bins)
                )
                result_matrix = mx.toarray()

        return result_matrix, is_empty

    @cachedmethod(cache=lambda s: s.block_intersection_cache,
                  key=lambda s, f, r, rb, cb: hashkey((r, rb.stripe_id, cb.stripe_id)),
                  lock=lambda s: s.block_intersection_cache_lock)
    def get_block_intersection_as_dense_matrix(
            self,
            f: h5py.File,
            resolution: np.int64,
            row_block: StripeDescriptor,
            col_block: StripeDescriptor
    ) -> np.ndarray:
        # y -> rows
        # x -> cols
        needs_transpose: bool = False
        if row_block.stripe_id > col_block.stripe_id:
            (row_block, col_block) = col_block, row_block
            needs_transpose = True

        mx_as_array: np.ndarray
        is_empty: bool

        mx_as_array, is_empty = self.load_saved_dense_block(f, resolution, row_block, col_block)

        result: np.ndarray

        if is_empty:
            result = mx_as_array
        else:
            # Block is square and lies on a diagonal:
            if row_block.stripe_id == col_block.stripe_id:
                mx_as_array = np.where(mx_as_array, mx_as_array, mx_as_array.T)

            if row_block.direction == ContigDirection.REVERSED:
                mx_as_array = np.flip(mx_as_array, axis=0)
            if col_block.direction == ContigDirection.REVERSED:
                mx_as_array = np.flip(mx_as_array, axis=1)

            result = mx_as_array

        if needs_transpose:
            result = result.T
        return result

    def get_rectangle(self, resolution: np.int64, x0_bins: np.int64, y0_bins: np.int64, x1_bins: np.int64,
                      y1_bins: np.int64) -> np.ndarray:
        """
        Returns dense submatrix for area [x0_bins, x1_bins) * [y0_bins, y1_bins), pixels are 0-indexed!
        Pixel coordinates are automatically constrained to lie between 0 and length at current resolution.
        """
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"

        length_bins_at_resolution: np.int64 = self.contig_tree.get_sizes()[0][resolution]
        x0_bins = constrain_coordinate(x0_bins, 0, length_bins_at_resolution)
        x1_bins = constrain_coordinate(x1_bins, 0, length_bins_at_resolution)
        y0_bins = constrain_coordinate(y0_bins, 0, length_bins_at_resolution)
        y1_bins = constrain_coordinate(y1_bins, 0, length_bins_at_resolution)

        x_zero_at_segment_bins: np.int64
        x_blocks: List[StripeDescriptor]
        x_zero_at_segment_bins, x_blocks = self.get_stripes_for_range(
            resolution,
            1 + x0_bins,
            x1_bins
        )

        y_zero_at_segment_bins: np.int64
        y_blocks: List[StripeDescriptor]
        y_zero_at_segment_bins, y_blocks = self.get_stripes_for_range(
            resolution,
            1 + y0_bins,
            y1_bins
        )

        if x0_bins >= x1_bins or y0_bins >= y1_bins:
            return np.zeros((0, 0))

        dense_x_length: np.int64 = x1_bins - x0_bins
        dense_y_length: np.int64 = y1_bins - y0_bins
        dense: np.ndarray = np.zeros((dense_x_length, dense_y_length), dtype=np.int32)

        # Map block id to the coordinate in resulting dense matrix, where its (0, 0) element would be located
        x_start_in_dense: Dict[np.int64, np.int64] = dict()
        y_start_in_dense: Dict[np.int64, np.int64] = dict()

        x_start_in_dense[x_blocks[0].stripe_id] = x_zero_at_segment_bins - x0_bins
        for i in range(1, len(x_blocks)):
            x_start_in_dense[x_blocks[i].stripe_id] = (
                x_start_in_dense[x_blocks[i - 1].stripe_id] + x_blocks[i - 1].stripe_length_bins
            )

        y_start_in_dense[y_blocks[0].stripe_id] = y_zero_at_segment_bins - y0_bins
        for i in range(1, len(y_blocks)):
            y_start_in_dense[y_blocks[i].stripe_id] = (
                y_start_in_dense[y_blocks[i - 1].stripe_id] + y_blocks[i - 1].stripe_length_bins
            )

        with self.hdf_file_lock.gen_rlock(), h5py.File(self.filepath, mode='r') as f:
            # Corner cases (literally):
            for x_block, y_block in itertools.product(
                    ([x_blocks[0], x_blocks[-1]] if len(x_blocks) > 1 else [x_blocks[0]]),
                    (
                            [y_blocks[0], y_blocks[-1]] if len(y_blocks) > 1 else [
                                y_blocks[0]])):
                x_0: np.int64 = x_start_in_dense[x_block.stripe_id]
                y_0: np.int64 = y_start_in_dense[y_block.stripe_id]
                block_intersection: np.ndarray = self.get_block_intersection_as_dense_matrix(
                    f,
                    resolution,
                    x_block,
                    y_block
                )

                assert (
                        block_intersection.shape == (x_block.stripe_length_bins, y_block.stripe_length_bins)
                ), "Wrong shape of intersection"

                block_intersection_submatrix_view: np.ndarray = \
                    block_intersection[
                        (-min(x_0, 0))
                        :
                        (
                            min(x_block.stripe_length_bins, dense_x_length - x_0)
                            if x_0 <= 0 else min((dense_x_length - x_0), x_block.stripe_length_bins)
                        ),
                        (-min(y_0, 0))
                        :
                        (
                            min(y_block.stripe_length_bins, dense_y_length - y_0)
                            if y_0 <= 0 else min((dense_y_length - y_0), y_block.stripe_length_bins)
                        )
                    ]

                if np.prod(block_intersection_submatrix_view.shape) != 0:
                    dense[max(0, x_0):min(dense_x_length, (x_0 + x_block.stripe_length_bins)),
                    max(0, y_0):min(dense_y_length, (y_0 + y_block.stripe_length_bins))
                    ] = block_intersection_submatrix_view

            # Vertical stripes:
            for x_block, y_block in itertools.product(
                    ([x_blocks[0], x_blocks[-1]] if len(x_blocks) > 1 else [x_blocks[0]]),
                    y_blocks[1:-1]
            ):
                x_0: np.int64 = x_start_in_dense[x_block.stripe_id]
                y_0: np.int64 = y_start_in_dense[y_block.stripe_id]
                block_intersection: np.ndarray = \
                    self.get_block_intersection_as_dense_matrix(f, resolution, x_block, y_block)

                assert (
                        block_intersection.shape == (x_block.stripe_length_bins, y_block.stripe_length_bins)
                ), "Wrong shape of intersection"

                block_intersection_submatrix_view: np.ndarray = \
                    block_intersection[
                        (-min(x_0, 0))
                        :
                        (min(x_block.stripe_length_bins, dense_x_length - x_0) if x_0 <= 0 else (dense_x_length - x_0)),
                        :
                    ]

                if np.prod(block_intersection_submatrix_view.shape) != 0:
                    dense[
                        max(0, x_0):min(dense_x_length, (x_0 + x_block.stripe_length_bins)),
                        y_0:(y_0 + y_block.stripe_length_bins)
                    ] = block_intersection_submatrix_view

            # Horizontal stripes:
            for x_block, y_block in itertools.product(
                    x_blocks[1:-1],
                    (
                            [y_blocks[0], y_blocks[-1]] if len(y_blocks) > 1 else [y_blocks[0]]
                    )
            ):
                x_0: np.int64 = x_start_in_dense[x_block.stripe_id]
                y_0: np.int64 = y_start_in_dense[y_block.stripe_id]
                block_intersection: np.ndarray = self.get_block_intersection_as_dense_matrix(
                    f,
                    resolution,
                    x_block,
                    y_block
                )

                assert (
                        block_intersection.shape == (x_block.stripe_length_bins, y_block.stripe_length_bins)
                ), "Wrong shape of intersection"

                block_intersection_submatrix_view: np.ndarray = \
                    block_intersection[
                        :,
                        (-min(y_0, 0))
                        :
                        (
                            min(y_block.stripe_length_bins, dense_y_length - y_0)
                            if y_0 <= 0 else (dense_y_length - y_0)
                        )
                    ]

                if np.prod(block_intersection_submatrix_view.shape) != 0:
                    dense[
                        x_0:(x_0 + x_block.stripe_length_bins),
                        max(0, y_0):min(dense_y_length, (y_0 + y_block.stripe_length_bins))
                    ] = block_intersection_submatrix_view

            # Frequent case (full cover):
            for x_block, y_block in itertools.product(x_blocks[1:-1], y_blocks[1:-1]):
                x_0: np.int64 = x_start_in_dense[x_block.stripe_id]
                y_0: np.int64 = y_start_in_dense[y_block.stripe_id]
                mx_flipped: np.ndarray = self.get_block_intersection_as_dense_matrix(f, resolution, x_block, y_block)
                assert (
                        mx_flipped.shape == (x_block.stripe_length_bins, y_block.stripe_length_bins)
                ), "Wrong shape of intersection"
                dense[
                    x_0:(x_0 + x_block.stripe_length_bins),
                    y_0:(y_0 + y_block.stripe_length_bins)
                ] = mx_flipped

        return dense

    def reverse_contig_by_id(self, contig_id: np.int64):
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"
        contig_descriptor, borders, index = self.contig_tree.get_contig_location(contig_id)
        for resolution in self.resolutions:
            (start_bins, end_bins) = borders[resolution]
            tree: StripeTree = self.matrix_trees[resolution]
            tree.reverse_direction_in_bins(1 + start_bins, end_bins)
        self.contig_tree.reverse_contigs_in_segment(index, index)
        self.clear_caches()

    def reverse_contigs_by_id(self, start_contig_id: np.int64, end_contig_id: np.int64):
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"
        start_contig_descriptor, borders_start, start_index = self.contig_tree.get_contig_location(start_contig_id)
        end_contig_descriptor, borders_end, end_index = self.contig_tree.get_contig_location(end_contig_id)

        if end_index < start_index:
            borders_start, start_index, borders_end, end_index = borders_end, end_index, borders_start, start_index

        ct_exposed_segment = self.contig_tree.expose_segment_by_count(start_index, end_index)
        if ct_exposed_segment.segment is not None:
            ct_exposed_segment.segment.reverse_subtree()
        self.contig_tree.commit_exposed_segment(ct_exposed_segment)

        for resolution in self.resolutions:
            mt = self.matrix_trees[resolution]
            (start_bins, end_bins) = (borders_start[resolution][0], borders_end[resolution][1])
            mt.reverse_direction_in_bins(1 + start_bins, end_bins)
        self.clear_caches()

    def get_contig_location(self, contig_id: np.int64) -> Tuple[
        ContigDescriptor,
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
        old_es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(target_order, target_order)

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
            target_scaffold_descriptor = self.scaffold_holder.get_scaffold_by_id(target_scaffold_id)
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
        target_es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(target_order, target_order)
        previous_contig_node: Optional[ContigTree.Node] = ContigTree.get_rightmost(target_es.less)
        target_contig_node: Optional[ContigTree.Node] = target_es.segment
        next_contig_node: Optional[ContigTree.Node] = ContigTree.get_leftmost(target_es.greater)
        self.contig_tree.commit_exposed_segment(target_es)

        def get_node_scaffold_descriptor(node: Optional[ContigTree.Node]) -> Optional[ScaffoldDescriptor]:
            return self.get_scaffold_by_id(node.contig_descriptor.scaffold_id) if node is not None else None

        previous_contig_scaffold: Optional[ScaffoldDescriptor] = get_node_scaffold_descriptor(previous_contig_node)
        target_contig_scaffold: Optional[ScaffoldDescriptor] = get_node_scaffold_descriptor(target_contig_node)
        next_contig_scaffold: Optional[ScaffoldDescriptor] = get_node_scaffold_descriptor(next_contig_node)

        scaffold_starts_at_target: bool = (
                (target_contig_scaffold is not None)
                and
                (
                        (previous_contig_scaffold is None)
                        or
                        (previous_contig_scaffold.scaffold_id != target_contig_scaffold.scaffold_id)
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
                        (target_contig_scaffold.scaffold_id != next_contig_scaffold.scaffold_id)
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

    def move_contig_by_id(
            self,
            contig_id: np.int64,
            target_order: np.int64,
            merge_into_existing_bordering_scaffold_if_exactly_one_is_present: bool = False,
    ) -> None:
        self.move_contigs_by_id(
            contig_id,
            contig_id,
            target_order,
            merge_into_existing_bordering_scaffold_if_exactly_one_is_present
        )

    def get_scaffold_by_id(self, scaffold_id: Optional[np.int64]) -> Optional[ScaffoldDescriptor]:
        return self.scaffold_housekeeping(scaffold_id) if scaffold_id is not None else None

    def scaffold_housekeeping(self, scaffold_id: np.int64) -> Optional[ScaffoldDescriptor]:
        try:
            scaffold_descriptor: ScaffoldDescriptor = self.scaffold_holder.get_scaffold_by_id(scaffold_id)
            scaffold_borders: Optional[ScaffoldBorders] = scaffold_descriptor.scaffold_borders
            if scaffold_borders is not None:
                start_contig_descriptor, _, _ = self.get_contig_location(scaffold_borders.start_contig_id)
                end_contig_descriptor, _, _ = self.get_contig_location(scaffold_borders.end_contig_id)
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
                    raise Exception(f"Only one bordering contig belongs to the scaffold with id={scaffold_id}?")
        except KeyError:
            return None

    def move_contigs_by_id(
            self,
            start_contig_id: np.int64,
            end_contig_id: np.int64,
            target_order: np.int64,
            merge_into_existing_bordering_scaffold_if_exactly_one_is_present: bool = False,
    ):
        assert self.state == ChunkedFile.FileState.OPENED, "Operation requires file to be opened"
        start_contig_descriptor, borders_start, start_order = self.contig_tree.get_contig_location(start_contig_id)
        if target_order == start_order:
            return

        start_old_scaffold_id: Optional[np.int64] = start_contig_descriptor.scaffold_id

        # Move borders of bordering scaffolds that are covered by this query:
        # Shorten scaffold of start contig if it is present:
        if start_old_scaffold_id is not None:
            start_old_scaffold: ScaffoldDescriptor = self.scaffold_holder.get_scaffold_by_id(start_old_scaffold_id)
            if start_old_scaffold.scaffold_borders.start_contig_id == start_contig_id:
                self.scaffold_holder.remove_scaffold_by_id(start_old_scaffold_id)
            else:
                start_es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(
                    start_order,
                    start_order
                )
                previous_of_start: Optional[ContigTree.Node] = ContigTree.get_rightmost(start_es.less)
                assert previous_of_start is not None, "Scaffold has not started but internal contig has no previous??"
                start_old_scaffold.scaffold_borders.end_contig_id = previous_of_start.contig_descriptor.contig_id
                self.contig_tree.commit_exposed_segment(start_es)

        end_contig_descriptor, borders_end, end_order = self.contig_tree.get_contig_location(end_contig_id)
        end_old_scaffold_id: Optional[np.int64] = end_contig_descriptor.scaffold_id

        # Shorten scaffold of the last contig if it is present:
        if end_old_scaffold_id is not None:
            end_old_scaffold: ScaffoldDescriptor = self.scaffold_holder.get_scaffold_by_id(end_old_scaffold_id)
            if end_old_scaffold.scaffold_borders.end_contig_id == end_contig_id:
                self.scaffold_holder.remove_scaffold_by_id(end_old_scaffold_id)
            else:
                end_es: ContigTree.ExposedSegment = self.contig_tree.expose_segment_by_count(
                    end_order,
                    end_order
                )
                next_of_end: Optional[ContigTree.Node] = ContigTree.get_leftmost(end_es.greater)
                assert next_of_end is not None, "Scaffold has not ended but internal node has no next??"
                end_old_scaffold.scaffold_borders.start_contig_id = next_of_end.contig_descriptor.contig_id
                self.contig_tree.commit_exposed_segment(end_es)

        if end_order < start_order:
            (
                start_contig_id,
                start_contig_descriptor,
                borders_start,
                start_order,
                end_contig_id,
                end_contig_descriptor,
                borders_end,
                end_order,
            ) = (
                end_contig_id,
                end_contig_descriptor,
                borders_end,
                end_order,
                start_contig_id,
                start_contig_descriptor,
                borders_start,
                start_order,
            )

        (
            ct_less,
            ct_segment,
            ct_greater
        ) = self.contig_tree.expose_segment_by_count(
            start_order,
            end_order
        )
        assert ct_segment is not None, "None encountered while moving contig segment??"
        ct_segment: ContigTree.Node
        ct_intermediate = self.contig_tree.merge_nodes(ct_less, ct_greater)

        self.contig_tree.root = ct_intermediate
        (
            target_previous_contig_scaffold,
            target_contig_scaffold,
            target_next_contig_scaffold,
            target_scaffold_starts_at_target,
            target_internal_scaffold_contig,
            target_scaffold_ends_at_target
        ) = self.check_scaffold_borders_at_position(target_order)

        if target_internal_scaffold_contig:
            # If moving inside existing scaffold -- merge into that scaffold
            ct_segment.contig_descriptor.scaffold_id = target_contig_scaffold.scaffold_id
        elif target_scaffold_starts_at_target and target_previous_contig_scaffold is not None:
            # If moving between two existing scaffolds -- create new scaffold from segment
            new_scaffold_descriptor: ScaffoldDescriptor = self.scaffold_holder.create_scaffold()
            ct_segment.contig_descriptor.scaffold_id = new_scaffold_descriptor.scaffold_id
        elif merge_into_existing_bordering_scaffold_if_exactly_one_is_present:
            # If moving to the position that has only one bordering scaffold -- merge into it:
            if target_previous_contig_scaffold is not None and target_contig_scaffold is None:
                ct_segment.contig_descriptor.scaffold_id = target_previous_contig_scaffold.scaffold_id
            elif target_previous_contig_scaffold is None and target_contig_scaffold is not None:
                ct_segment.contig_descriptor.scaffold_id = target_contig_scaffold.scaffold_id

        (ct_new_less, ct_new_greater) = self.contig_tree.split_node_by_count(
            ct_intermediate,
            target_order
        )
        if ct_new_less is not None:
            ct_new_less_length, _ = ct_new_less.get_sizes()
        else:
            ct_new_less_length: Dict[np.int64, np.int64] = {0: 0}
            for resolution in self.resolutions:
                ct_new_less_length[resolution] = 0
        ct_new_less_with_segment = self.contig_tree.merge_nodes(ct_new_less, ct_segment)
        self.contig_tree.root = self.contig_tree.merge_nodes(ct_new_less_with_segment, ct_new_greater)

        for resolution in self.resolutions:
            mt = self.matrix_trees[resolution]
            (start_bins, end_bins) = (borders_start[resolution][0], borders_end[resolution][1])
            (mt_less, mt_segment, mt_greater) = mt.expose_segment(1 + start_bins, end_bins)

            mt_intermediate = mt.merge_nodes(mt_less, mt_greater)
            (mt_new_less, mt_new_greater) = mt.split_node_by_length(
                mt_intermediate,
                ct_new_less_length[resolution],
                True
            )

            mt_new_less_with_segment = mt.merge_nodes(mt_new_less, mt_segment)
            mt.root = mt.merge_nodes(mt_new_less_with_segment, mt_new_greater)
        self.clear_caches()

    def dump_stripe_info(self, f: h5py.File):
        for resolution in self.resolutions:
            stripe_info_group: h5py.Group = f[f'/resolutions/{resolution}/contigs']
            stripe_count: np.int64 = self.matrix_trees[resolution].get_node_count()

            ordered_stripe_ids: h5py.Dataset = create_dataset_if_not_exists(
                'ordered_stripe_ids', stripe_info_group, shape=(stripe_count,),
                maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args
            )

            stripe_direction: h5py.Dataset = create_dataset_if_not_exists(
                'stripe_direction', stripe_info_group, shape=(stripe_count,),
                maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
            )

            ordered_stripe_ids_backup: h5py.Dataset = create_dataset_if_not_exists(
                'ordered_stripe_ids_backup', stripe_info_group, shape=(stripe_count,),
                maxshape=(None,), dtype=np.int64, **additional_dataset_creation_args
            )

            stripe_direction_backup: h5py.Dataset = create_dataset_if_not_exists(
                'stripe_direction_backup', stripe_info_group, shape=(stripe_count,),
                maxshape=(None,), dtype=np.int8, **additional_dataset_creation_args
            )

            get_attribute_value_or_create_if_not_exists('stripe_backup_done', False, stripe_info_group)

            if get_attribute_value_or_create_if_not_exists('stripe_write_finished', False, stripe_info_group):
                stripe_info_group.attrs['stripe_backup_done'] = False
                stripe_info_group.attrs['stripe_write_finished'] = False

            stripe_write_finished: bool = stripe_info_group.attrs.get('stripe_write_finished')
            assert not stripe_write_finished, "Incorrect state of writing changes?"

            stripe_backup_done: bool = stripe_info_group.attrs.get('stripe_backup_done')

            if not stripe_backup_done:
                ordered_stripe_ids_backup[:] = ordered_stripe_ids[:]
                stripe_direction_backup[:] = stripe_direction[:]
                stripe_info_group.attrs['stripe_backup_done'] = True

            ordered_stripe_ids_list: List[np.int64] = []
            stripe_direction_list: np.ndarray = np.ones(shape=(stripe_count,), dtype=np.int8)

            def traversal_fn(node: StripeTree.Node) -> None:
                stripe_descriptor: StripeDescriptor = node.stripe_descriptor
                stripe_id: np.int64 = stripe_descriptor.stripe_id
                ordered_stripe_ids_list.append(stripe_id)
                stripe_direction_list[stripe_id] = stripe_descriptor.direction.value

            self.matrix_trees[resolution].traverse(traversal_fn)

            ordered_stripe_ids[:] = ordered_stripe_ids_list[:]
            stripe_direction[:] = stripe_direction_list[:]

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
        resolution_to_contig_hide_type_backup: Dict[np.int64, h5py.Dataset] = dict()

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

        get_attribute_value_or_create_if_not_exists('contig_backup_done', False, contig_info_group)

        if get_attribute_value_or_create_if_not_exists('contig_write_finished', False, contig_info_group):
            contig_info_group.attrs['contig_backup_done'] = False
            contig_info_group.attrs['contig_write_finished'] = False

        contig_write_finished: bool = contig_info_group.attrs.get('contig_write_finished')
        assert not contig_write_finished, "Incorrect state of writing changes?"

        contig_backup_done: bool = contig_info_group.attrs.get('contig_backup_done')

        if not contig_backup_done:
            ordered_contig_ids_backup[:] = ordered_contig_ids[:]
            contig_direction_backup[:] = contig_direction[:]
            contig_scaffold_id_backup[:] = contig_scaffold_id[:]
            for resolution in self.resolutions:
                resolution_to_contig_hide_type[resolution][:] = resolution_to_contig_hide_type_backup[resolution][:]
            contig_info_group.attrs['contig_backup_done'] = True

        ordered_contig_ids_list: List[np.int64] = []
        contig_direction_list: np.ndarray = np.ones(shape=(contig_count,), dtype=np.int8)
        # ContigId -> [Resolution -> ContigHideType]
        contig_hide_types: List[Dict[np.int64, ContigHideType]] = [None] * contig_count
        contig_old_scaffold_id: np.ndarray = np.zeros(shape=(contig_count,), dtype=np.int64)
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
        scaffold_new_id_to_old_id: np.ndarray = np.zeros(shape=(len(used_scaffold_ids),), dtype=np.int64)
        for new_id, old_id in enumerate(used_scaffold_ids):
            scaffold_old_id_to_new_id[old_id] = new_id
            scaffold_new_id_to_old_id[new_id] = old_id

        ordered_contig_ids[:] = ordered_contig_ids_list[:]
        contig_direction[:] = contig_direction_list[:]
        contig_scaffold_id[:] = list(map(lambda old_s_id: scaffold_old_id_to_new_id[old_s_id] if old_s_id != -1 else -1,
                                         contig_old_scaffold_id))[:]

        for resolution in self.resolutions:
            contig_id_to_contig_hide_type_at_resolution = np.ones(shape=(contig_count,), dtype=np.int8)
            for contig_id, contig_presence_in_resolution in enumerate(contig_hide_types):
                contig_id_to_contig_hide_type_at_resolution[contig_id] = contig_presence_in_resolution[resolution].value
            resolution_to_contig_hide_type[resolution][:] = contig_id_to_contig_hide_type_at_resolution[:]

        contig_info_group.attrs['contig_write_finished'] = True

        return scaffold_new_id_to_old_id

    def dump_scaffold_info(
            self,
            f: h5py.File,
            scaffold_new_id_to_old_id: np.ndarray,
    ) -> None:
        scaffold_info_group: h5py.Group = create_group_if_not_exists('scaffold_info', f['/'])
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

        get_attribute_value_or_create_if_not_exists('scaffold_backup_done', False, scaffold_info_group)

        if get_attribute_value_or_create_if_not_exists('scaffold_write_finished', False, scaffold_info_group):
            scaffold_info_group.attrs['scaffold_backup_done'] = False
            scaffold_info_group.attrs['scaffold_write_finished'] = False

        scaffold_write_finished: bool = scaffold_info_group.attrs.get('scaffold_write_finished')
        assert not scaffold_write_finished, "Incorrect state of writing changes?"

        scaffold_backup_done: bool = scaffold_info_group.attrs.get('scaffold_backup_done')

        if not scaffold_backup_done:
            scaffold_name_backup_ds[:] = scaffold_name_ds[:]
            scaffold_start_backup_ds[:] = scaffold_start_ds[:]
            scaffold_end_backup_ds[:] = scaffold_end_ds[:]
            scaffold_direction_backup_ds[:] = scaffold_direction_ds[:]
            scaffold_spacer_backup_ds[:] = scaffold_spacer_ds[:]
            scaffold_info_group.attrs['scaffold_backup_done'] = True

        scaffold_names: List[str] = []
        scaffold_starts: np.ndarray = np.zeros(shape=(scaffold_count,), dtype=np.int64)
        scaffold_ends: np.ndarray = np.zeros(shape=(scaffold_count,), dtype=np.int64)
        scaffold_directions: np.ndarray = np.zeros(shape=(scaffold_count,), dtype=np.int8)
        scaffold_spacers: np.ndarray = np.zeros(shape=(scaffold_count,), dtype=np.int32)

        for new_id, old_id in enumerate(scaffold_new_id_to_old_id):
            scaffold_descriptor: ScaffoldDescriptor = self.scaffold_holder.get_scaffold_by_id(old_id)
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
        try:
            with self.hdf_file_lock.gen_wlock(), h5py.File(self.filepath, mode='a') as f:
                self.dump_stripe_info(f)
                f.flush()
                scaffold_new_id_to_old_id = self.dump_contig_info(f)
                f.flush()
                self.dump_scaffold_info(f, scaffold_new_id_to_old_id)
                f.flush()
                self.clear_caches(saved_blocks=True)
        except Exception as e:
            print(f"Exception was thrown during save process: {str(e)}\nFile might be saved incorrectly.")
            self.state = ChunkedFile.FileState.INCORRECT
            raise e

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

            ordered_finalization_records: List[Tuple[FinalizeRecordType, List[ContigDescriptor]]] = []

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
                    if contig_descriptor.scaffold_id == last_scaffold_id:
                        assert (
                                ordered_finalization_records[-1][0] == FinalizeRecordType.SCAFFOLD
                        ), "Last contig descriptor has a scaffold_id but marked as out-of-scaffold?"
                        ordered_finalization_records[-1][1].append(contig_descriptor)
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
