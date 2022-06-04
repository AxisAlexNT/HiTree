import copy
import datetime
import math

import h5py
from h5py import Dataset
from scipy.sparse import coo_matrix

from hitree.core.common import ContigDirection, ContigHideType
from hitree.core.stripe_tree import *


def save_values_to_group(
        blocks: List[Tuple[List[np.int64], List[np.int64], List]],
        leftmost_block_index: np.int64,
        block_datasets: Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, Dataset],
        current_sparse_offset: np.int64,
        current_dense_offset: np.int64,
        stripes: List[StripeDescriptor],
        dense_submatrix_size: np.int64,
) -> Tuple[np.int64, np.int64]:
    (block_rows, block_cols, block_vals, block_offset, block_length, dense_blocks) = block_datasets
    block_count_in_row: np.int64 = len(blocks)
    row_stripe: StripeDescriptor = stripes[leftmost_block_index]
    block_row_count: np.int64 = row_stripe.stripe_length_bins
    for block_index in range(leftmost_block_index, block_count_in_row):
        col_stripe: StripeDescriptor = stripes[block_index]
        block = blocks[block_index]
        block_index_in_datasets: np.int64 = leftmost_block_index * block_count_in_row + block_index
        # Do not store empty blocks:
        block_nonzero_element_count = len(block[0])
        if block_nonzero_element_count == 0:
            # print(f'Not dumping empty block {block_index}')
            block_length[block_index_in_datasets] = 0
            continue

        block_col_count: np.int64 = col_stripe.stripe_length_bins

        block_length[block_index_in_datasets] = block_nonzero_element_count

        if (
                (
                        block_row_count == dense_submatrix_size
                )
                and
                (
                        block_col_count == dense_submatrix_size
                )
                and
                (
                        block_nonzero_element_count >= ((block_row_count * block_col_count) // 2)
                )
        ):
            # Save as a dense matrix
            block_offset[block_index_in_datasets] = -current_dense_offset - 1
            dense_blocks.resize(1 + current_dense_offset, axis=0)
            mx_coo = coo_matrix(
                (
                    block[2],
                    (
                        block[0],
                        block[1]
                    )
                ),
                shape=(block_row_count, block_col_count)
            )
            dense_blocks[current_dense_offset, 0, :, :] = mx_coo.toarray()
            current_dense_offset += 1
        else:
            # Save as sparse matrix
            block_offset[block_index_in_datasets] = current_sparse_offset
            block_rows[current_sparse_offset:current_sparse_offset + block_nonzero_element_count] = block[0]
            block_cols[current_sparse_offset:current_sparse_offset + block_nonzero_element_count] = block[1]
            block_vals[current_sparse_offset:current_sparse_offset + block_nonzero_element_count] = block[2]
            current_sparse_offset += block_nonzero_element_count
    return current_sparse_offset, current_dense_offset


def clear_blocks(blocks: List[Tuple[List, List, List]], current_row_stripe_id: np.int64):
    for block_index in range(current_row_stripe_id, len(blocks)):
        blocks[block_index][0].clear()
        blocks[block_index][1].clear()
        blocks[block_index][2].clear()


def dump_stripe_data(
        src_file: h5py.File,
        dst_file: h5py.File,
        submatrix_size: np.int64,
        resolution: np.int64,
        contig_id_to_contig_length_bp: np.ndarray,
        path_to_name_and_length: str = '/chroms/',
        additional_dataset_creation_args: Optional[dict] = None
) -> List[StripeDescriptor]:
    assert len(src_file[f'{path_to_name_and_length}/length']) == len(src_file[f'{path_to_name_and_length}/name'])
    stripes_group: h5py.Group = dst_file.create_group(f'/resolutions/{resolution}/stripes')
    contig_start_bins: h5py.Dataset = src_file[f'/resolutions/{resolution}/indexes/chrom_offset'].astype(np.int64)
    contig_length_bins_ds: np.ndarray = contig_start_bins[1:] - contig_start_bins[:-1]

    stripe_descriptors: List[StripeDescriptor] = []
    stripe_index: np.int64 = 0

    # Split contigs into blocks
    contigs_count: np.int64 = len(contig_length_bins_ds)
    for contig_id in range(0, contigs_count):
        contig_length_bins: np.int64 = contig_length_bins_ds[contig_id]
        contig_part_count: np.int64 = math.ceil(contig_length_bins / submatrix_size)
        contig_length_bp: np.int64 = contig_id_to_contig_length_bp[contig_id]

        stripe_descriptors.extend((
            StripeDescriptor.make_stripe_descriptor(
                stripe_index + contig_part_id,
                min(submatrix_size, contig_length_bins - contig_part_id * submatrix_size),
                min(submatrix_size * resolution, contig_length_bp - contig_part_id * submatrix_size * resolution),
                StripeDirection.FORWARD,
                contig_id
            )

            for contig_part_id in range(0, contig_part_count)
        ))

        stripe_index += contig_part_count

    if additional_dataset_creation_args is None:
        additional_dataset_creation_args = {}
    stripes_group.create_dataset('ordered_stripe_ids', data=[stripe.stripe_id for stripe in stripe_descriptors],
                                 dtype=np.int64, **additional_dataset_creation_args)
    stripes_group.create_dataset('stripe_length_bins',
                                 data=[stripe.stripe_length_bins for stripe in stripe_descriptors], dtype=np.int64,
                                 **additional_dataset_creation_args)
    stripes_group.create_dataset('stripe_length_bp', data=[stripe.stripe_length_bp for stripe in stripe_descriptors],
                                 dtype=np.int64, **additional_dataset_creation_args)
    stripes_group.create_dataset('stripe_direction',
                                 data=[stripe.direction.value for stripe in stripe_descriptors],
                                 **additional_dataset_creation_args)
    stripes_group.create_dataset('stripes_contig_id',
                                 data=[stripe.contig_id for stripe in stripe_descriptors], dtype=np.int64,
                                 **additional_dataset_creation_args)
    return stripe_descriptors


def dump_contig_data(
        src_file: h5py.File,
        dst_file: h5py.File,
        path_to_name_and_length: str,
        resolutions: List[np.int64],
        additional_dataset_creation_args: Optional[dict] = None
) -> np.ndarray:
    anyresolution: np.int64 = resolutions[0]
    contig_info_group: h5py.Group = dst_file.create_group(f'/contig_info/')
    contig_info_group.copy(src_file[f'{path_to_name_and_length}/name'], 'contig_name')

    contig_count: np.int64 = len(contig_info_group['contig_name'])

    src_contig_chrom_offset: h5py.Dataset = src_file[f'/resolutions/{anyresolution}/indexes/chrom_offset']
    src_bin_ends: h5py.Dataset = src_file[f'/resolutions/{anyresolution}/bins/end']

    if additional_dataset_creation_args is None:
        additional_dataset_creation_args = {}

    contig_id_to_contig_length_bp: np.ndarray

    if 'length' in src_file[f'/resolutions/{anyresolution}/chroms'].keys():
        contig_info_group.copy(src_file[f'/resolutions/{anyresolution}/chroms/length'], 'contig_length_bp')
        contig_id_to_contig_length_bp = src_file[f'/resolutions/{anyresolution}/chroms/length'][:]
    else:
        contig_length: np.ndarray = np.zeros(contig_count, dtype=np.int64)

        contig_length[:-1] = src_bin_ends[src_contig_chrom_offset[1:-1] - 1]
        contig_length[-1] = src_bin_ends[-1]

        contig_info_group.create_dataset('contig_length_bp', data=contig_length, dtype=np.int64,
                                         **additional_dataset_creation_args)
        contig_id_to_contig_length_bp = contig_length[:]

    contig_info_group.create_dataset('contig_direction',
                                     data=[ContigDirection.FORWARD.value for _ in range(0, contig_count)],
                                     **additional_dataset_creation_args)
    contig_info_group.create_dataset('ordered_contig_ids', data=list(range(0, contig_count)),
                                     dtype=np.int64, **additional_dataset_creation_args)
    contig_info_group.create_dataset('contig_scaffold_id', data=[-1 for _ in range(0, contig_count)],
                                     dtype=np.int64, **additional_dataset_creation_args)

    for resolution in resolutions:
        contigs_group: h5py.Group = dst_file.create_group(f'/resolutions/{resolution}/contigs')
        contig_start_bins: h5py.Dataset = src_file[f'/resolutions/{resolution}/indexes/chrom_offset'].astype(np.int64)
        contig_length_bins_ds: np.ndarray = contig_start_bins[1:] - contig_start_bins[:-1]
        contigs_group.create_dataset('contig_length_bins', data=contig_length_bins_ds, dtype=np.int64,
                                     **additional_dataset_creation_args)
        contigs_group.create_dataset('contig_hide_type',
                                     data=[ContigHideType.AUTO_SHOWN.value for _ in range(0, contig_count)],
                                     dtype=np.int8, **additional_dataset_creation_args)

    return contig_id_to_contig_length_bp


def is_sorted(a: np.ndarray) -> bool:
    return np.all(a[:-1] <= a[1:])


def convert_cooler_into_flat_hitree_model(
        src_file_path: str,
        dst_file_path: str,
        get_name_and_length_path: Callable[[np.int64], str],
        resolutions: Optional[List[np.int64]] = None,
        additional_dataset_creation_args: Optional[dict] = None
):
    submatrix_size: np.int64 = 256
    hdf5_max_chunk_size: np.int64 = 32 * 1024 * 1024 * 8
    max_fetch_size: np.int64 = 8 * 1024 * 1024
    if additional_dataset_creation_args is None:
        additional_dataset_creation_args = {}

    with h5py.File(name=src_file_path, mode='r') as src_file:
        if resolutions is None:
            resolutions = [np.int64(sdn) for sdn in filter(lambda s: s.isnumeric(), src_file['resolutions'].keys())]
        with h5py.File(name=dst_file_path, mode='w') as dst_file:
            contig_id_to_contig_length_bp: np.ndarray = dump_contig_data(
                src_file,
                dst_file,
                get_name_and_length_path(resolutions[0]),
                resolutions,
                additional_dataset_creation_args
            )
            for resolution in sorted(resolutions, reverse=True):
                print(f"Resolution {resolution} out of {resolutions}")
                stripes = dump_stripe_data(
                    src_file,
                    dst_file,
                    submatrix_size,
                    resolution,
                    contig_id_to_contig_length_bp,
                    get_name_and_length_path(resolution),
                    additional_dataset_creation_args
                )
                res_group: h5py.Group = dst_file.create_group(f'resolutions/{resolution}/treap_coo')
                res_group.attrs.create(name='dense_submatrix_size', data=submatrix_size)
                res_group.attrs.create(name='hdf5_max_chunk_size', data=hdf5_max_chunk_size)
                src_bins_count = len(src_file[f'resolutions/{resolution}/bins/end'])
                src_pixels = src_file[f'resolutions/{resolution}/pixels']
                src_pixel_row: h5py.Dataset = src_pixels['bin1_id']
                src_pixel_col: h5py.Dataset = src_pixels['bin2_id']
                src_pixel_val: h5py.Dataset = src_pixels['count']
                nonzero_pixel_count: np.int64 = len(src_pixel_row)
                res_group.attrs.create(name='bins_count', data=src_bins_count)
                # Maximum count of stripes horizontally [covering rows 0..submatrix_size)
                stripes_count = len(stripes)
                stripe_data: List[
                    Tuple[
                        List[np.int64],
                        List[np.int64],
                        List]
                ] = [([], [], []) for _ in range(0, stripes_count)]
                # Note: since stripe !== contig, these are in fact different:
                stripe_start_indices: np.ndarray = np.cumsum([0] + [s.stripe_length_bins for s in stripes],
                                                             dtype=np.int64)
                all_rows_start_indices: h5py.Dataset = src_file[f'resolutions/{resolution}/indexes/bin1_offset']

                fetch_size: np.int64 = len(src_pixel_row)
                fetchblock_count: np.int64 = fetch_size // max_fetch_size + min(1, fetch_size % max_fetch_size)

                print(
                    f"bins_count: {src_bins_count}, "
                    f"stripes_count: {stripes_count}, "
                    f"fetch_size: {fetch_size}, "
                    f"fetchblock_count: {fetchblock_count}"
                )

                current_row_stripe_id: np.int64 = 0
                current_row_stripe_start_row: np.int64 = 0
                # Note: these values are exclusive:
                current_row_stripe_max_row: np.int64 = stripes[0].stripe_length_bins
                pixeltable_start_position: np.int64 = 0

                ds_creation_args: dict = copy.deepcopy(additional_dataset_creation_args)
                ds_creation_args['chunks'] = True

                block_rows: h5py.Dataset = res_group.create_dataset(
                    'block_rows', shape=(nonzero_pixel_count,),
                    dtype=np.int64,
                    **ds_creation_args
                )
                block_cols: h5py.Dataset = res_group.create_dataset(
                    'block_cols', shape=(nonzero_pixel_count,),
                    dtype=np.int64,
                    **ds_creation_args
                )
                block_vals: h5py.Dataset = res_group.create_dataset(
                    'block_vals', shape=(nonzero_pixel_count,),
                    dtype=src_pixel_val.dtype,
                    **ds_creation_args
                )
                total_block_count: np.int64 = stripes_count * stripes_count
                block_offset: h5py.Dataset = res_group.create_dataset(
                    'block_offset', shape=(total_block_count,),
                    dtype=np.int64,
                    **ds_creation_args
                )
                block_length: h5py.Dataset = res_group.create_dataset(
                    'block_length', shape=(total_block_count,),
                    dtype=np.int64,
                    **ds_creation_args
                )

                dense_blocks: h5py.Dataset = res_group.create_dataset(
                    'dense_blocks',
                    shape=(1, 1, submatrix_size, submatrix_size),
                    maxshape=(None, 1, submatrix_size, submatrix_size),
                    dtype=src_pixel_val.dtype,
                    **ds_creation_args,
                )

                block_datasets: Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, Dataset] = (
                    block_rows, block_cols, block_vals, block_offset, block_length, dense_blocks)

                res_group.attrs.create(name='stripes_count', data=stripes_count)

                current_sparse_offset: np.int64 = 0
                current_dense_offset: np.int64 = 0

                for pixel_row_fetchblock, pixel_col_fetchblock, pixel_cnt_fetchblock in (
                        (
                                src_pixel_row[i * max_fetch_size: min((1 + i) * max_fetch_size, fetch_size)],
                                src_pixel_col[i * max_fetch_size: min((1 + i) * max_fetch_size, fetch_size)],
                                src_pixel_val[i * max_fetch_size: min((1 + i) * max_fetch_size, fetch_size)]
                        ) for i in range(0, fetchblock_count)
                ):
                    minimum_row_in_fetchblock: np.int64 = pixel_row_fetchblock[0]
                    maximum_row_in_fetchblock: np.int64 = pixel_row_fetchblock[-1]
                    row_start_indices: np.ndarray = all_rows_start_indices[
                                                    minimum_row_in_fetchblock:2 + maximum_row_in_fetchblock
                                                    ] - pixeltable_start_position
                    if row_start_indices[0] < 0:
                        first_zero_index: np.int64 = np.searchsorted(row_start_indices, 0, side='left')
                        row_start_indices[:first_zero_index] = 0
                        pass
                    pixeltable_start_position += len(pixel_row_fetchblock)

                    for i in range(0, len(row_start_indices) - 1):
                        row_start_index: np.int64 = row_start_indices[i]
                        row_number: np.int64 = pixel_row_fetchblock[row_start_index]
                        next_row_start_index: np.int64 = row_start_indices[1 + i]
                        # print(f"Parsing row {row_number}")
                        while row_number >= current_row_stripe_max_row:
                            current_sparse_offset, current_dense_offset = save_values_to_group(
                                stripe_data,
                                current_row_stripe_id,
                                block_datasets,
                                current_sparse_offset,
                                current_dense_offset,
                                stripes,
                                submatrix_size
                            )
                            clear_blocks(stripe_data, 0)
                            dst_file.flush()
                            current_row_stripe_id += 1
                            current_row_stripe_start_row = current_row_stripe_max_row
                            current_row_stripe_max_row += stripes[current_row_stripe_id].stripe_length_bins
                            print(
                                f"{print(datetime.datetime.now())}\nStripe {current_row_stripe_id} out of {stripes_count}")
                        if row_start_index == next_row_start_index:
                            continue
                        assert row_start_index >= 0
                        assert next_row_start_index >= 0
                        single_row_row: np.ndarray = pixel_row_fetchblock[
                                                     row_start_index:next_row_start_index] - current_row_stripe_start_row
                        single_row_col: np.ndarray = pixel_col_fetchblock[row_start_index:next_row_start_index]
                        single_row_cnt: np.ndarray = pixel_cnt_fetchblock[row_start_index:next_row_start_index]
                        assert max(single_row_row) == min(single_row_row)
                        assert is_sorted(
                            single_row_col), "Column indices inside one row must follow in the sorted order"
                        block_start_indices: np.ndarray = np.searchsorted(
                            single_row_col,
                            stripe_start_indices[current_row_stripe_id:],
                            side='left'
                        )

                        for j in range(0, len(block_start_indices) - 1):
                            block_in_row_start_index: np.int64 = block_start_indices[j]
                            next_block_in_row_start_index: np.int64 = block_start_indices[1 + j]
                            # If there are some points from current block in current row:
                            if next_block_in_row_start_index > block_in_row_start_index:
                                row_ids = single_row_row[block_in_row_start_index:next_block_in_row_start_index]
                                assert 0 <= max(row_ids) < submatrix_size
                                assert 0 <= min(row_ids) < submatrix_size
                                stripe_data[current_row_stripe_id + j][0].extend(row_ids)
                                column_ids_original = single_row_col[
                                                      block_in_row_start_index:next_block_in_row_start_index
                                                      ]
                                column_ids_in_block = column_ids_original - stripe_start_indices[
                                    current_row_stripe_id + j]
                                assert 0 <= max(column_ids_in_block) < submatrix_size
                                assert 0 <= min(column_ids_in_block) < submatrix_size
                                stripe_data[current_row_stripe_id + j][1].extend(column_ids_in_block)
                                stripe_data[current_row_stripe_id + j][2].extend(
                                    single_row_cnt[block_in_row_start_index:next_block_in_row_start_index])

                current_sparse_offset, current_dense_offset = save_values_to_group(
                    stripe_data,
                    current_row_stripe_id,
                    block_datasets,
                    current_sparse_offset,
                    current_dense_offset,
                    stripes,
                    submatrix_size
                )
                clear_blocks(stripe_data, current_row_stripe_id)
                dst_file.flush()
