import multiprocessing
import multiprocessing.managers
from typing import List, Tuple
import copy
from pathlib import Path
from typing import NamedTuple, Optional
from flask import scaffold

import numpy as np
from HiCT_Library.hict.core.scaffold_tree import ScaffoldTree

from hict.core.chunked_file import ChunkedFile
from hict.core.common import ContigDescriptor, ContigDirection, QueryLengthUnit, ScaffoldDescriptor, ScaffoldDirection
from hict.core.contig_tree import ContigTree


class ContactMatrixFacet(object):
    """
    This facet is designed to be the main API object to interact with our files and model without using the model methods directly.
    """

    class IncorrectFileStateError(Exception):
        """
        General exception that indicates file or model are in the incorrect state and require attention.
        """
        pass

    class IncorrectResolution(Exception):
        """
        An exception that indicates wrong resolution was queried (in a sense that it is not stored in file).
        """
        pass

    @staticmethod
    def get_file_descriptor(
        filepath: str,
        block_cache_size: int = 64,
        multithreading_pool_size: int = 8,
        mp_manager: Optional[multiprocessing.managers.SyncManager] = None
    ) -> ChunkedFile:
        """
        Create descriptor for working with files in our format.

        :param filepath: Path to the file relative to your working directory.
        :param block_cache_size: Size of cache for dense blocks (each at most max_dense_size*max_dense_size*sizeof(dtype) bytes).
        :return: File descriptor.
        """
        f: ChunkedFile = ChunkedFile(
            filepath,
            block_cache_size,
            multithreading_pool_size=multithreading_pool_size,
            mp_manager=mp_manager
        )
        return f

    @staticmethod
    def open_file(f: ChunkedFile) -> None:
        """
        Open file using file descriptor and build initial state of the model.

        :param f: File descriptor.
        """
        if f.state == ChunkedFile.FileState.CLOSED:
            f.open()
        elif f.state == ChunkedFile.FileState.INCORRECT:
            raise ContactMatrixFacet.IncorrectFileStateError()

    @staticmethod
    def close_file(f: ChunkedFile, need_save: bool = True) -> None:
        """
        Close file using descriptor and save pending changes.

        :param f: File descriptor.
        :param need_save: Whether to store unsaved changes in file or not.
        """
        if f.state == ChunkedFile.FileState.OPENED:
            f.close(need_save=need_save)
        elif f.state == ChunkedFile.FileState.INCORRECT:
            raise ContactMatrixFacet.IncorrectFileStateError()

    @staticmethod
    def save_changes(f: ChunkedFile) -> None:
        """
        Save pending changes into the file using its descriptor.

        :param f: File descriptor.
        """
        if f.state == ChunkedFile.FileState.OPENED:
            f.save()
        elif f.state == ChunkedFile.FileState.INCORRECT:
            raise ContactMatrixFacet.IncorrectFileStateError()

    @staticmethod
    def get_resolutions_list(f: ChunkedFile) -> List[np.int64]:
        """
        Gets a list of resolutions that are stored in the given chunked file. File should be opened.

        :param f: File descriptor.
        """
        if f.state == ChunkedFile.FileState.OPENED:
            return copy.deepcopy(f.resolutions)
        else:
            raise ContactMatrixFacet.IncorrectFileStateError()

    @staticmethod
    def get_matrix_size_bins(f: ChunkedFile, resolution: np.int64) -> np.int64:
        """
        Returns contact matrix size at the given resolution in bins. File should be opened.

        :param f: File descriptor.
        :param resolution: Resolution at which the contact matrix size is queried.
        """
        if f.state == ChunkedFile.FileState.OPENED:
            if resolution not in f.resolutions:
                raise ContactMatrixFacet.IncorrectResolution()
            return (
                (
                    f.contig_tree.root.get_sizes()[0][resolution]
                ) if f.contig_tree.root is not None else 0
            )
        else:
            raise ContactMatrixFacet.IncorrectFileStateError()

    @staticmethod
    def get_matrix_size_px(f: ChunkedFile, resolution: np.int64) -> np.int64:
        """
        Returns contact matrix size at the given resolution in pixels. File should be opened.

        :param f: File descriptor.
        :param resolution: Resolution at which the contact matrix size is queried.
        """
        tree = f.contig_tree
        assert (
            tree is not None
        ), "Contig tree is not present?"
        if f.state == ChunkedFile.FileState.OPENED:
            if resolution not in f.resolutions:
                raise ContactMatrixFacet.IncorrectResolution()
            return (
                (
                    tree.root.get_sizes()[2][resolution]
                ) if tree.root is not None else 0
            )
        else:
            raise ContactMatrixFacet.IncorrectFileStateError()

    class BasePairInPixelPosition(NamedTuple):
        """
        A tuple that describes queried position in both bp and pixels.
        """
        resolution: np.int64
        query_position_bp: np.int64
        intra_contig_position_bp: np.int64
        intra_contig_position_bins: np.int64
        global_position_px: np.int64
        global_position_bins: np.int64
        less_segment_length_px: np.int64
        less_segment_length_bins: np.int64
        greater_segment_length_px: np.int64
        greater_segment_length_bins: np.int64

    @staticmethod
    def get_px_by_bp(f: ChunkedFile, x0_bp: np.int64, resolution: np.int64 = 0) -> BasePairInPixelPosition:
        """
        Queries position of a given base pair in resolution.

        :param f: File descriptor.
        :param x0_bp: Position expressed in base pairs.
        :param resolution: Resolution for which the pixel is queried.
        :return: Position of a pixel which corresponds to the given base pair.
        """
        ct = f.contig_tree
        assert (
            ct is not None
        ), "Contig tree is not present?"
        es_x0: ContigTree.ExposedSegment = ct.expose_segment_by_length(
            x0_bp, x0_bp, 0)
        x0_in_contig_position_bp = x0_bp - \
            (es_x0.less.get_sizes()[0][0] if es_x0.less is not None else 0)
        x0_in_contig_position_bins = (
            x0_in_contig_position_bp - 1) // resolution
        ls_size_bins: np.int64 = es_x0.less.get_sizes(
        )[0][resolution] if es_x0.less is not None else 0
        x0_position_bins: np.int64 = ls_size_bins + x0_in_contig_position_bins
        ls_size_px: np.int64 = es_x0.less.get_sizes(
        )[2][resolution] if es_x0.less is not None else 0
        x0_position_px: np.int64 = ls_size_px + x0_in_contig_position_bins
        result = ContactMatrixFacet.BasePairInPixelPosition(
            resolution=resolution,
            query_position_bp=x0_bp,
            intra_contig_position_bp=x0_in_contig_position_bp,
            intra_contig_position_bins=x0_in_contig_position_bins,
            global_position_px=x0_position_px,
            global_position_bins=x0_position_bins,
            less_segment_length_bins=ls_size_bins,
            less_segment_length_px=ls_size_px,
            greater_segment_length_bins=es_x0.greater.get_sizes(
            )[0][resolution] if es_x0.greater is not None else 0,
            greater_segment_length_px=es_x0.greater.get_sizes(
            )[2][resolution] if es_x0.greater is not None else 0
        )
        ct.commit_exposed_segment(es_x0)
        return result

    @staticmethod
    def get_dense_submatrix(
            f: ChunkedFile,
            resolution: np.int64,
            x0: np.int64,
            y0: np.int64,
            x1: np.int64,
            y1: np.int64,
            units: QueryLengthUnit = QueryLengthUnit.PIXELS,
            exclude_hidden_contigs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetches requested area from contact matrix in the given resolution.

        :param f: File descriptor.
        :param resolution: Experiment resolution.
        :param x0: Start column of query expressed in given units (inclusive).
        :param y0: Start row of query expressed in given units (inclusive).
        :param x1: End column of query expressed in given units (exclusive).
        :param y1: End row of query expressed in given units (exclusive).
        :param units: Either QueryLengthUnit.PIXELS (0-indexed) or QueryLengthUnit.BASE_PAIRS (1-indexed). In both cases borders are inclusive.
        :return: A tuple of (M, w_r, w_c) where M is dense 2D numpy array which contains contact map submatrix for the given region, w_r is row bin weights and w_c is column bin weights.
        """
        # x0 = max(0, x0)
        # x1 = max(0, x1)
        # y0 = max(0, y0)
        # y1 = max(0, y1)

        # if x0 > x1:
        #     x0, x1 = x1, x0
        # if y0 > y1:
        #     y0, y1 = y1, y0

        if resolution not in f.resolutions:
            raise ContactMatrixFacet.IncorrectResolution()
        # submatrix_and_weights: Tuple[np.ndarray, np.ndarray, np.ndarray]
        if units == QueryLengthUnit.BASE_PAIRS:
            # (x|y)(0|1)_bp -> (x|y)(0|1)_px using 1:1 "resolution" to find start and ending contigs
            # In start contig, find which bin (pixel) this query falls into,
            # by dividing in-contig length by a resolution
            # The same goes with the end contig
            # Use subsize of left and right segments to subtract bp from their length in bp

            x0_in_contig_px = ContactMatrixFacet.get_px_by_bp(
                f, x0, resolution)
            x1_in_contig_px = ContactMatrixFacet.get_px_by_bp(
                f, x1, resolution)
            y0_in_contig_px = ContactMatrixFacet.get_px_by_bp(
                f, y0, resolution)
            y1_in_contig_px = ContactMatrixFacet.get_px_by_bp(
                f, y1, resolution)

            submatrix_and_weights = f.get_submatrix(
                resolution,
                x0_in_contig_px.global_position_px,
                y0_in_contig_px.global_position_px,
                1 + x1_in_contig_px.global_position_px,
                1 + y1_in_contig_px.global_position_px,
                exclude_hidden_contigs=exclude_hidden_contigs
            )
        else:
            # submatrix = f.get_submatrix(resolution, x0, y0, 1 + x1, 1 + y1, units, exclude_hidden_contigs)
            submatrix_and_weights = f.get_submatrix(
                resolution,
                x0, y0,
                x1, y1,
                exclude_hidden_contigs=(exclude_hidden_contigs or (
                    units == QueryLengthUnit.PIXELS))
            )

        return submatrix_and_weights

    @staticmethod
    def apply_cooler_balance_to_dense_matrix(
        dense_matrix: np.ndarray,
        row_weights: np.ndarray,
        col_weights: np.ndarray,
        inplace: bool = False
    ) -> np.ndarray:
        result: np.ndarray = dense_matrix if inplace else np.copy(dense_matrix)
        result = result * col_weights
        result = (result.T * row_weights).T
        return result

    @staticmethod
    def reverse_selection_range_bp(f: ChunkedFile, start_bp: np.int64, end_bp: np.int64) -> None:
        """
        Performs reversal of contig segment between given start and end bps (both are extended to cover the whole contig). Changes orientation of each contig on that segment and reverses their order. If any scaffold intersects with selection range, then selection range is extended to include this scaffold.

        :param f: File descriptor.
        :param start_bp: Left border (in assembly base pairs).
        :param end_bp: Right border (in assembly base pairs).
        """
        f.reverse_selection_range_bp(start_bp, end_bp)

    @staticmethod
    def move_selection_range_bp(f: ChunkedFile, start_bp: np.int64, end_bp: np.int64, target_start_bp: np.int64) -> None:
        """
        Moves contig segment between given start and end bps (both are extended to cover the whole contig). If any scaffold intersects with selection range, then selection range is extended to include this scaffold.

        :param f: File descriptor.
        :param start_bp: Left border (in assembly base pairs).
        :param end_bp: Right border (in assembly base pairs).
        :param target_start_bp: Rightmost possible position for start of moved segment.
        """
        f.move_selection_range_bp(
            start_bp, end_bp, target_start_bp)

    @staticmethod
    def group_selection_range_into_scaffold(f: ChunkedFile, start_bp: np.int64, end_bp: np.int64, name: Optional[str] = None, spacer_length: int = 1000) -> None:
        """
        Groups segment between given bps into the new scaffold, both bordering contigs are included in it. All scaffolds that intersect with given segment, would be fully added into the new scaffold (so its borders might actually be different from start_contig_id and end_contig_id).

        :param f: File descriptor.
        :param start_bp: Left border (in assembly base pairs).
        :param end_bp: Right border (in assembly base pairs).
        :param name: New scaffold's name. If not provided, would be generaed automatically.
        :param spacer_length: How many spacers 'N' to include in the final FASTA assembly on the borders of this scaffold.
        """
        f.scaffold_segment(start_bp, end_bp, name, spacer_length)

    @staticmethod
    def ungroup_selection_range(f: ChunkedFile, start_bp: np.int64, end_bp: np.int64) -> None:
        """
        Takes selection range between given bps and removes all scaffolds that intersect with it.

        :param f: File descriptor.
        :param start_bp: Left border (in assembly base pairs).
        :param end_bp: Right border (in assembly base pairs).
        """
        f.unscaffold_segment(start_bp, end_bp)

    @staticmethod
    def load_assembly_from_agp(f: ChunkedFile, agp_filepath: Path) -> None:
        """
        Loads assembly from specified AGP file.

        :param f: HiCT File descriptor.
        :param agp_filename: Path to the AGP file.
        """
        assert agp_filepath.exists() and agp_filepath.is_file(
        ), "AGP file path should point to existent file"
        f.load_assembly_from_agp(agp_filepath)
        
    @staticmethod
    def get_ordered_contigs(f: ChunkedFile) -> List[Tuple[ContigDescriptor, ContigDirection]]:
        """
        Returns the list of contigs together with their directions in the current assembly order.

        :param f: File descriptor.
        :return A list of tuples `(ctg, dir)`.
        """
        tree = f.contig_tree
        result: List[Tuple[ContigDescriptor, ContigDirection]]
        
        assert (
            tree is not None
        ), "No contig tree is present?"
        
        def traverse_fn(n: ContigTree.Node) -> None:
            result.append((
                n.contig_descriptor,
                n.true_direction()
            ))
        
        with tree.root_lock.gen_rlock():
            tree.traverse(traverse_fn)
            
        return result
    
    @staticmethod
    def get_ordered_scaffolds(f: ChunkedFile) -> List[Tuple[Optional[ScaffoldDescriptor], int]]:
        """
        Returns the list of scaffolds together with their lengths in base pairs in the current assembly order. Also included are unscaffolded regions which have length provided but the descriptor is `None`. If you build prefix sum of the lengths, you'll find each scaffold's borders in base pairs.

        :param f: File descriptor.
        :return A list of tuples `(scaf, len)` where `scaf` can be either a `ScaffoldDescriptor` or `None` (for unscaffolded region) and `len` is its length in base pairs.
        """
        tree = f.scaffold_tree      
        result: List[Tuple[Optional[ScaffoldDescriptor], int]]
        
        assert (
            tree is not None
        ), "No scaffold tree is present?"
        
        def traverse_fn(n: ScaffoldTree.Node) -> None:
            result.append((
                n.scaffold_descriptor,
                int(n.length_bp)
            ))
        
        with tree.root_lock.gen_rlock():
            tree.traverse(traverse_fn)
            
        return result
    
    @staticmethod
    def get_assembly_info(f: ChunkedFile) -> Tuple[List[Tuple[ContigDescriptor, ContigDirection]], List[Tuple[Optional[ScaffoldDescriptor], int]]]:
        """
        Returns a pair of ordered contig descriptors and scaffold descriptors. Check docs `get_ordered_contigs` and `get_ordered_scaffolds` for the return value description.

        :param f: File descriptor.
        :return A tuple of list of ordered contig descriptots and a list of ordered scaffold descriptors.
        """
        
        contig_tree = f.contig_tree      
        scaffold_tree = f.scaffold_tree      
                
        assert (
            contig_tree is not None
        ), "Contig tree is None?"
        assert (
            scaffold_tree is not None
        ), "Scaffold tree is None?"
        
        
        with contig_tree.root_lock.gen_rlock(), scaffold_tree.root_lock.gen_rlock():
            return ContactMatrixFacet.get_ordered_contigs(f), ContactMatrixFacet.get_ordered_scaffolds(f)
            
        
