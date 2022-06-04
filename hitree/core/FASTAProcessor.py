import io
from typing import Tuple, List, Dict, Optional

import numpy as np
from Bio import SeqIO

from hitree.core.common import ContigDirection, ScaffoldDescriptor, ContigDescriptor, FinalizeRecordType


class FASTAProcessor(object):

    def __init__(
            self,
            filename: str
    ) -> None:
        self.records = dict()
        self.initial_ctg_ord: list[str, ...] = []
        for record in SeqIO.parse(filename, 'fasta'):
            self.initial_ctg_ord.append(record.id)
            self.records[record.id] = record
        self.initial_ctg_ord = tuple(self.initial_ctg_ord)

    def list_records(self) -> Tuple[str, ...]:
        return tuple(s for s in self.initial_ctg_ord)

    def get_dna_string_for_single_contig(
            self,
            contig_name: str,
            contig_direction: ContigDirection,
            offset_from_start: np.int64 = 0,
            offset_before_end: np.int64 = 0
    ) -> str:
        dna_seq = self.records[contig_name].seq
        l = len(dna_seq) - 1
        if contig_direction == ContigDirection.FORWARD:
            return str(dna_seq[offset_from_start: l - offset_before_end])
        elif contig_direction == ContigDirection.REVERSED:
            return str(dna_seq.reverse_complement()[offset_from_start: l - offset_before_end])
        else:
            raise Exception(f"Incorrect contig direction: {str(contig_direction.name)}={str(contig_direction.value)}")

    def get_dna_string_for_multiple_contigs_inside_scaffold(
            self,
            scaffold_descriptor: ScaffoldDescriptor,
            ordered_contig_descriptors: List[ContigDescriptor],
            contig_id_to_contig_name: List[str],
    ) -> str:
        ctg_count: int = len(ordered_contig_descriptors)
        if ctg_count <= 0:
            raise Exception("Contig count must be positive")
        elif ctg_count == 1:
            return self.get_dna_string_for_single_contig(
                contig_id_to_contig_name[ordered_contig_descriptors[0].contig_id],
                ordered_contig_descriptors[0].direction,
            )
        else:
            spacer_str: str = 'N' * scaffold_descriptor.spacer_length
            return spacer_str.join(
                self.get_dna_string_for_single_contig(
                    contig_id_to_contig_name[ordered_contig_descriptors[i].contig_id],
                    ordered_contig_descriptors[i].direction,
                ) for i in range(ctg_count)
            )

    def get_fasta_record_for_scaffold(
            self,
            scaffold_descriptor: ScaffoldDescriptor,
            ordered_contig_descriptors: List[ContigDescriptor],
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
            contig_name: str
    ) -> str:
        out_str = ''
        out_str += f'>{contig_name}\n'
        out_str += self.get_dna_string_for_single_contig(
            contig_name,
            contig_descriptor.direction
        )
        out_str += '\n'
        return out_str

    def finalize_fasta_for_assembly(
            self,
            writable_stream,
            ordered_finalization_records: List[Tuple[FinalizeRecordType, List[ContigDescriptor]]],
            scaffold_id_to_scaffold_descriptor: Dict[np.int64, ScaffoldDescriptor],
            contig_id_to_contig_name: List[str]
    ):
        for record_order, finalization_record in enumerate(ordered_finalization_records):
            record_type = finalization_record[0]
            if record_type == FinalizeRecordType.CONTIG_NOT_IN_SCAFFOLD:
                assert (
                    len(finalization_record[1])
                ), "Finalization record for single contig not in scaffold must contain only one contig descriptor"
                contig_descriptor: ContigDescriptor = finalization_record[1][0]
                if isinstance(writable_stream, io.BytesIO):
                    writable_stream.write(self.get_fasta_record_for_single_contig_not_in_scaffold(
                        contig_descriptor,
                        contig_id_to_contig_name[contig_descriptor.contig_id],
                    ).encode('utf-8'))
                elif isinstance(writable_stream, io.StringIO):
                    print(
                        self.get_fasta_record_for_single_contig_not_in_scaffold(
                            contig_descriptor,
                            contig_id_to_contig_name[contig_descriptor.contig_id],
                        ),
                        end='',
                        file=writable_stream
                    )
                else:
                    raise Exception("Cannot write stream")
            elif record_type == FinalizeRecordType.SCAFFOLD:
                assert (
                        len(finalization_record[1]) > 0
                ), "Finalization record for scaffold must contain at least one contig"
                scaffold_id: Optional[np.int64] = finalization_record[1][0].scaffold_id
                assert scaffold_id is not None, "Finalization record for scaffold must have non-None scaffold_id"
                assert all(map(
                    (lambda cd: cd.scaffold_id == scaffold_id),
                    finalization_record[1]
                )), "All contigs inside one scaffold record must have the same scaffold ids"
                scaffold_descriptor: ScaffoldDescriptor = scaffold_id_to_scaffold_descriptor[scaffold_id]
                assert (
                        scaffold_descriptor.scaffold_borders is not None
                ), "Scaffold that has contigs in it must have defined borders"
                bordering_contig_id: List[np.int64] = [
                    finalization_record[1][0].contig_id,
                    finalization_record[1][-1].contig_id
                ]
                assert (
                        scaffold_descriptor.scaffold_borders.start_contig_id in bordering_contig_id
                ), "Scaffold starting contig id must be in scaffold record"
                assert (
                        scaffold_descriptor.scaffold_borders.end_contig_id in bordering_contig_id
                ), "Scaffold ending contig id must be in scaffold record"
                if isinstance(writable_stream, io.BytesIO):
                    writable_stream.write(self.get_fasta_record_for_scaffold(
                        scaffold_descriptor,
                        finalization_record[1],
                        contig_id_to_contig_name
                    ).encode('utf-8'))
                elif isinstance(writable_stream, io.StringIO):
                    print(
                        self.get_fasta_record_for_scaffold(
                            scaffold_descriptor,
                            finalization_record[1],
                            contig_id_to_contig_name
                        ),
                        end='',
                        file=writable_stream
                    )
                else:
                    raise Exception("Cannot write stream")
            else:
                raise Exception(
                    f"Unknown finalization record type: {record_type.name}={record_type.value} at position {record_order}"
                )
