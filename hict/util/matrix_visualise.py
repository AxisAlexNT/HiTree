import string
import numpy as np
from hict.api.ContactMatrixFacet import ContactMatrixFacet
from hict.core.chunked_file import ChunkedFile
import matplotlib.colors as clr

from hict.core.common import ScaffoldDescriptor


class MatrixVisualise(object):
    @staticmethod
    def get_matrix(chunk_file_with_agp: ChunkedFile, contig_name: string, resolution: int,
                   weighted: bool) -> np.ndarray:
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

        matrix = matrix.astype('float32')
        matrix[matrix == 0] = np.nan

        return matrix

    @staticmethod
    def log_matrix(matrix: np.ndarray, log_base: float = 10, addition: float = 1,
                   remove_zeros: bool = True) -> np.ndarray:
        if remove_zeros:
            matrix[matrix == 0] = np.NaN
        return np.log(matrix + addition) / np.log(log_base)

    @staticmethod
    def get_colormap(start_color_hex: string, mid_color_hex: string, end_color_hex: string,
                     gradient_levels: tuple = (0, 0.5, 1)) -> clr.LinearSegmentedColormap:
        if len(gradient_levels) != 3:
            raise Exception("gradient_levels tuple should have 3 elements")
        return clr.LinearSegmentedColormap.from_list('custom colormap', [(gradient_levels[0], start_color_hex),
                                                                         (gradient_levels[1], mid_color_hex),
                                                                         (gradient_levels[2], end_color_hex)])

    @staticmethod
    def get_colormap_diverging(first_quarter: float = 0.250, second_quarter: float = 0.750,
                               start_color:tuple = (0.000, 0.145, 0.702), end_color:tuple =  (0.780, 0.012, 0.051),
                               mid_color:tuple =  (1.000, 1.000, 1.000) ) \
            -> clr.LinearSegmentedColormap:
        return clr.LinearSegmentedColormap.from_list('diverging_clr', (
            (0.000, start_color),
            (first_quarter, start_color),
            (0.500,mid_color),
            (second_quarter, end_color),
            (1.000, end_color)))

    @staticmethod
    def calculate_diag_means(matrix: np.ndarray, scaffold_first: ScaffoldDescriptor,
                             scaffold_second: ScaffoldDescriptor, res: string = 'exp/obs') -> np.ndarray:
        result = np.zeros_like(matrix, dtype='float64')
        expected = np.zeros_like(matrix, dtype='float64')
        if scaffold_first.scaffold_id == scaffold_second.scaffold_id:
            n = len(matrix)
            averages_at_dist = [np.nanmean([matrix[i, j - d] for i, j in zip(range(d, n), range(d, n))]) for d in
                                range(n)]
            for i in range(n):
                for j in range(n):
                    expected[i, j] = averages_at_dist[abs(i - j)]
        else:
            averages_at_dist = np.nanmean(matrix)
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    expected[i, j] = averages_at_dist[abs(i - j)]

        if res == 'exp/obs':
            return expected/matrix

        if res == 'exp':
            return expected

        if res == 'exp-obs':
            return expected - matrix

        if res == 'obs-exp':
            return matrix - expected

        if res == 'obs/exp':
            return matrix/expected

        return result
