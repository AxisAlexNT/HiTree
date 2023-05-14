from typing import Tuple, List, Dict, Optional

import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from hict.core.common import ContigDirection, ScaffoldDescriptor, ContigDescriptor, FinalizeRecordType


class FASTAProcessor(object):

    initial_ctg_ord: Tuple[str, ...]

    def __init__(
            self,
            filename: str
    ) -> None:
        self.records: Dict[str, SeqRecord] = dict()
        initial_ctg_ord_list: List[str] = []
        for record in SeqIO.parse(filename, 'fasta'):
            initial_ctg_ord_list.append(record.id)
            self.records[record.id] = record
        self.initial_ctg_ord: Tuple[str, ...] = tuple(initial_ctg_ord_list)

    def list_records(self) -> Tuple[str, ...]:
        return tuple(s for s in self.initial_ctg_ord)

    def get_fasta_for_range(
            self, file_like, ctg_list: List[Tuple[ContigDescriptor, ContigDirection]],
            out_fasta_header: str,
            offset_from_start_bp: int = 0,
            offset_from_end_bp: int = 0,
            intercontig_spacer: str = 500*'N',
    ) -> None:
        out_sequence_list: List[str] = list()
        last_ctg_order: int = len(ctg_list) - 1
        for ctg_order, (ctg, ctg_dir) in enumerate(ctg_list):
            ctg_name: str = ctg.contig_name
            s_offset: int = 0
            e_offset: int = 0
            if ctg_order == 0:
                s_offset = offset_from_start_bp
            if ctg_order == last_ctg_order:
                e_offset = offset_from_end_bp
            out_sequence_list.append(
                self.get_dna_string_for_single_contig(
                    ctg_name,
                    ctg_dir,
                    s_offset, e_offset
                )
            )
        out_sequence = intercontig_spacer.join(out_sequence_list)
        out_record: bytes = f">{out_fasta_header}\n{out_sequence}".encode(
            encoding='utf-8')
        file_like.write(out_record)

    def get_dna_string_for_single_contig(
            self,
            contig_name: str,
            contig_direction: ContigDirection,
            offset_from_start: int = 0,
            offset_before_end: int = 0
    ) -> str:
        dna_seq = self.records[contig_name].seq
        if contig_direction == ContigDirection.FORWARD:
            return str(dna_seq[offset_from_start: (-offset_before_end if (offset_before_end > 0) else None)])
        elif contig_direction == ContigDirection.REVERSED:
            rc_seq = dna_seq.reverse_complement()
            return str(rc_seq[offset_from_start: (-offset_before_end if (offset_before_end > 0) else None)])
        else:
            raise Exception(
                f"Incorrect contig direction: {str(contig_direction.name)}={str(contig_direction.value)}")

    def get_dna_string_for_multiple_contigs_inside_scaffold(
            self,
            scaffold_descriptor: ScaffoldDescriptor,
            ordered_contig_descriptors: List[Tuple[ContigDescriptor, ContigDirection]],
            contig_id_to_contig_name: List[str],
    ) -> str:
        ctg_count: int = len(ordered_contig_descriptors)
        if ctg_count <= 0:
            raise Exception("Contig count must be positive")
        elif ctg_count == 1:
            return self.get_dna_string_for_single_contig(
                contig_id_to_contig_name[ordered_contig_descriptors[0][0].contig_id],
                ordered_contig_descriptors[0][1]
            )
        else:
            spacer_str: str = 'N' * scaffold_descriptor.spacer_length
            return spacer_str.join(
                self.get_dna_string_for_single_contig(
                    contig_id_to_contig_name[ctg_descr.contig_id],
                    ctg_dir
                ) for ctg_descr, ctg_dir in ordered_contig_descriptors
            )

    def get_fasta_record_for_scaffold(
            self,
            scaffold_descriptor: ScaffoldDescriptor,
            ordered_contig_descriptors: List[Tuple[ContigDescriptor, ContigDirection]],
            contig_id_to_contig_name: List[str],
    ) -> str:
        out_str = f">{scaffold_descriptor.scaffold_name}\n"
        out_str += self.get_dna_string_for_multiple_contigs_inside_scaffold(
            scaffold_descriptor,
            ordered_contig_descriptors,
            contig_id_to_contig_name
        )
        out_str += '\n'
        return out_str

    def get_fasta_record_for_single_contig_not_in_scaffold(
            self,
            contig_descriptor: ContigDescriptor,
            contig_direction: ContigDirection,
            contig_name: Optional[str] = None
    ) -> str:
        if contig_name is None:
            contig_name = contig_descriptor.contig_name
        out_str = ''
        out_str += f'>{contig_name}\n'
        out_str += self.get_dna_string_for_single_contig(
            contig_name,
            contig_direction
        )
        out_str += '\n'
        return out_str

    def finalize_fasta_for_assembly(
            self,
            file_like,
            ordered_finalization_records: List[Tuple[FinalizeRecordType, List[Tuple[ContigDescriptor, ContigDirection]]]],
            scaffold_id_to_scaffold_descriptor: Dict[np.int64, ScaffoldDescriptor],
            contig_id_to_contig_name: Optional[List[str]] = None
    ):
        for record_order, finalization_record in enumerate(ordered_finalization_records):
            record_type = finalization_record[0]
            if record_type == FinalizeRecordType.CONTIG_NOT_IN_SCAFFOLD:
                assert (
                    len(finalization_record[1])
                ), "Finalization record for single contig not in scaffold must contain only one contig descriptor"
                contig_descriptor, contig_direction = finalization_record[1][0]
                contig_record_str: str = self.get_fasta_record_for_single_contig_not_in_scaffold(
                    contig_descriptor,
                    contig_direction,
                    contig_id_to_contig_name[contig_descriptor.contig_id] if contig_id_to_contig_name is not None else None,
                )
                contig_record_bytes: bytes = contig_record_str.encode(
                    encoding='utf-8')
                #print(bytes(contig_record_bytes), end=None, file=file_like)
                file_like.write(contig_record_bytes)
            elif record_type == FinalizeRecordType.SCAFFOLD:
                raise Exception("Not yet implemented") #TODO: fix new scaffold output
                assert (
                    len(finalization_record[1]) > 0
                ), "Finalization record for scaffold must contain at least one contig"
                scaffold_id: Optional[np.int64] = finalization_record[1][0][0].scaffold_id
                assert scaffold_id is not None, "Finalization record for scaffold must have non-None scaffold_id"
                scaffold_descriptor: ScaffoldDescriptor = scaffold_id_to_scaffold_descriptor[
                    scaffold_id]
                scaffold_record_str: str = self.get_fasta_record_for_scaffold(
                    scaffold_descriptor,
                    finalization_record[1],
                    contig_id_to_contig_name
                )
                scaffold_record_bytes: bytes = scaffold_record_str.encode(
                    encoding='utf-8')
                #print(scaffold_record_bytes, end=None, file=file_like)
                file_like.write(scaffold_record_bytes)
            else:
                raise Exception(
                    f"Unknown finalization record type: {record_type.name}={record_type.value} at position {record_order}"
                )
