from typing import NamedTuple

import numpy as np

from hitree.core.chunked_file import ChunkedFile
from hitree.core.common import LengthUnit
from hitree.core.contig_tree import ContigTree


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
    def get_file_descriptor(filepath: str) -> ChunkedFile:
        """
        Create descriptor for working with files in our format.

        :param filepath: Path to the file relative to your working directory.
        :return: File descriptor.
        """
        f: ChunkedFile = ChunkedFile(filepath)
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

    class BasePairInPixelPosition(NamedTuple):
        """
        A tuple that describes queried position in both bp and pixels.
        """
        resolution: np.int64
        query_position_bp: np.int64
        intra_contig_position_bp: np.int64
        intra_contig_position_px: np.int64
        global_position_px: np.int64
        less_segment_length_px: np.int64
        greater_segment_length_px: np.int64

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
        es_x0: ContigTree.ExposedSegment = ct.expose_segment_by_length(x0_bp, x0_bp, 0)
        x0_in_contig_position_bp = x0_bp - (es_x0.less.get_sizes()[0][0] if es_x0.less is not None else 0)
        x0_in_contig_position_px = (x0_in_contig_position_bp - 1) // resolution
        ls_size_px: np.int64 = es_x0.less.get_sizes()[0][resolution] if es_x0.less is not None else 0
        x0_position_px: np.int64 = ls_size_px + x0_in_contig_position_px
        result = ContactMatrixFacet.BasePairInPixelPosition(
            resolution,
            x0_bp,
            x0_in_contig_position_bp,
            x0_in_contig_position_px,
            x0_position_px,
            ls_size_px,
            es_x0.greater.get_sizes()[0][resolution] if es_x0.greater is not None else 0
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
            units: LengthUnit = LengthUnit.PIXELS
    ) -> np.ndarray:
        """
        Fetches requested area from contact matrix in the given resolution.

        :param f: File descriptor.
        :param resolution: Experiment resolution.
        :param x0: Start column of query expressed in given units.
        :param y0: Start row of query expressed in given units.
        :param x1: End column of query expressed in given units.
        :param y1: End row of query expressed in given units.
        :param units: Either LengthUnit.PIXELS (0-indexed) or LengthUnit.BASE_PAIRS (1-indexed). In both cases borders are inclusive.
        :return: Dense 2D numpy array which contains contact map submatrix for the given region.
        """
        # x0 = max(0, x0)
        # x1 = max(0, x1)
        # y0 = max(0, y0)
        # y1 = max(0, y1)

        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        if resolution not in f.resolutions:
            raise ContactMatrixFacet.IncorrectResolution()
        if units == LengthUnit.PIXELS:
            return f.get_rectangle(resolution, x0, y0, 1 + x1, 1 + y1)
        elif units == LengthUnit.BASE_PAIRS:
            # (x|y)(0|1)_bp -> (x|y)(0|1)_px using 1:1 "resolution" to find start and ending contigs
            # In start contig, find which bin (pixel) this query falls into,
            # by dividing in-contig length by a resolution
            # The same goes with the end contig
            # Use subsize of left and right segments to subtract bp from their length in bp

            x0_in_contig_px = ContactMatrixFacet.get_px_by_bp(f, x0, resolution)
            x1_in_contig_px = ContactMatrixFacet.get_px_by_bp(f, x1, resolution)
            y0_in_contig_px = ContactMatrixFacet.get_px_by_bp(f, y0, resolution)
            y1_in_contig_px = ContactMatrixFacet.get_px_by_bp(f, y1, resolution)

            # Return dense submatrix:
            return f.get_rectangle(
                resolution,
                x0_in_contig_px.global_position_px,
                y0_in_contig_px.global_position_px,
                1 + x1_in_contig_px.global_position_px,
                1 + y1_in_contig_px.global_position_px
            )
        else:
            raise Exception("Incorrect measurement unit")

    @staticmethod
    def reverse_contigs_between_given_ids(f: ChunkedFile, start_contig_id: np.int64, end_contig_id: np.int64) -> None:
        """
        Performs reversal of contig segment between given start and end contigs (both are included). Changes orientation of each contig on that segment and reverses their order.

        :param f: File descriptor.
        :param start_contig_id: ID of the left bordering contig.
        :param end_contig_id: ID of the right bordering contig.
        """
        f.reverse_contigs_by_id(start_contig_id, end_contig_id)

    @staticmethod
    def reverse_contig_by_id(f: ChunkedFile, contig_id: np.int64) -> None:
        """
        Changes orientation of the contig with given ID.

        :param f: File descriptor.
        :param contig_id: ID of contig to be reversed.
        """
        f.reverse_contig_by_id(contig_id)

    @staticmethod
    def move_contigs_by_id(f: ChunkedFile, start_contig_id: np.int64, end_contig_id: np.int64,
                           target_order: np.int64) -> None:
        """
        Takes a segment of contigs and moves it so that the left bordering contig now resides at the target position in assembly. Both bordering contigs are included into the segment.

        :param f: File descriptor.
        :param start_contig_id: ID of the left segment bordering contig.
        :param end_contig_id: ID of the right segment bordering contig.
        :param target_order: Desired position of the start contig after move operation.
        """
        f.move_contigs_by_id(start_contig_id, end_contig_id, target_order)

    @staticmethod
    def move_contig_by_id(f: ChunkedFile, contig_id: np.int64, target_order: np.int64) -> None:
        """
        Moves contig with given ID to the desired position in assembly.

        :param f: File descriptor.
        :param contig_id: ID of contig to be moved.
        :param target_order: Desired position of given contig after move operation.
        """
        f.move_contig_by_id(contig_id, target_order)
