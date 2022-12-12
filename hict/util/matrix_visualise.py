import string
import numpy as np
from hict.api.ContactMatrixFacet import ContactMatrixFacet
from hict.core.chunked_file import ChunkedFile
import matplotlib.colors as clr


class MatrixVisualise(object):
    @staticmethod
    def get_matrix(chunk_file_with_agp: ChunkedFile, contig_name: string, resolution: int, weighted: bool) -> np.ndarray:
        _, contig_location, contig_location_exclude_hidden, _ = chunk_file_with_agp.get_contig_location(
            chunk_file_with_agp.contig_name_to_contig_id[contig_name])
        matrix, w_r, w_c = ContactMatrixFacet.get_dense_submatrix(chunk_file_with_agp,
                                                                  resolution,
                                                                  contig_location[resolution][0],
                                                                  contig_location[resolution][0],
                                                                  contig_location[resolution][1],
                                                                  contig_location[resolution][1],
                                                                  exclude_hidden_contigs=False,
                                                                  fetch_cooler_weights=weighted)

        if weighted:
            matrix = ContactMatrixFacet.apply_cooler_balance_to_dense_matrix(matrix, w_r, w_c, False)
        return matrix

    @staticmethod
    def log_matrix(matrix: np.ndarray, log_base: float = 10, addition: float = 1, remove_zeros: bool = True) -> np.ndarray:
        if remove_zeros:
            matrix[matrix == 0] = np.NaN
        return np.log(matrix+addition)/np.log(log_base)

    @staticmethod
    def get_colormap(start_color_hex: string, mid_color_hex: string, end_color_hex: string,
                     gradient_levels: tuple = (0, 0.5, 1)) -> clr.LinearSegmentedColormap:
        if len(gradient_levels) != 3:
            raise Exception("gradient_levels tuple should have 3 elements")
        return clr.LinearSegmentedColormap.from_list('custom colormap', [(gradient_levels[0], start_color_hex),
                                                                         (gradient_levels[1], mid_color_hex),
                                                                         (gradient_levels[2], end_color_hex)])
