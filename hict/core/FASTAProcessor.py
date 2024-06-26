#  MIT License
#
#  Copyright (c) 2021-2024. Aleksandr Serdiukov, Anton Zamyatin, Aleksandr Sinitsyn, Vitalii Dravgelis and Computer Technologies Laboratory ITMO University team.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
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
            s_offset: int = 0
            e_offset: int = 0
            if ctg_order == 0:
                s_offset = offset_from_start_bp
            if ctg_order == last_ctg_order:
                e_offset = offset_from_end_bp
            out_sequence_list.append(
                self.get_cropped_dna_string_for_single_contig(
                    ctg,
                    ctg_dir,
                    s_offset,
                    e_offset
                )
            )
        out_sequence = intercontig_spacer.join(out_sequence_list)
        out_record: bytes = f">{out_fasta_header}\n{out_sequence}".encode(
            encoding='utf-8')
        file_like.write(out_record)

    def get_cropped_dna_string_for_single_contig(
        self,
        ctg: ContigDescriptor,
        ctg_dir: ContigDirection,
        offset_from_start: int = 0,
        offset_before_end: int = 0
    ) -> str:
        base_dna_seq = self.records[ctg.contig_name_in_source_fasta]
        contig_dna_seq = base_dna_seq[ctg.offset_inside_fasta_contig:(
            ctg.offset_inside_fasta_contig+ctg.contig_length_at_resolution[0])]
        if ctg_dir == ContigDirection.FORWARD:
            return str(contig_dna_seq[offset_from_start: (-offset_before_end if (offset_before_end > 0) else None)].seq)
        elif ctg_dir == ContigDirection.REVERSED:
            rc_seq = contig_dna_seq.reverse_complement()
            return str(rc_seq[offset_from_start: (-offset_before_end if (offset_before_end > 0) else None)].seq)
        else:
            raise Exception(
                f"Incorrect contig direction: {str(ctg_dir.name)}={str(ctg_dir.value)}"
            )

    def get_dna_string_for_multiple_contigs_inside_scaffold(
            self,
            scaffold_descriptor: ScaffoldDescriptor,
            ordered_contig_descriptors: List[Tuple[ContigDescriptor, ContigDirection]],
    ) -> str:
        ctg_count: int = len(ordered_contig_descriptors)
        if ctg_count <= 0:
            raise Exception("Contig count must be positive")
        elif ctg_count == 1:
            return self.get_cropped_dna_string_for_single_contig(
                ordered_contig_descriptors[0][0],
                ordered_contig_descriptors[0][1]
            )
        else:
            spacer_str: str = 'N' * scaffold_descriptor.spacer_length
            return spacer_str.join(
                self.get_cropped_dna_string_for_single_contig(
                    ctg_descr,
                    ctg_dir
                ) for ctg_descr, ctg_dir in ordered_contig_descriptors
            )

    def get_fasta_record_for_scaffold(
            self,
            scaffold_descriptor: ScaffoldDescriptor,
            ordered_contig_descriptors: List[Tuple[ContigDescriptor, ContigDirection]],
    ) -> str:
        out_str = f">{scaffold_descriptor.scaffold_name}\n"
        out_str += self.get_dna_string_for_multiple_contigs_inside_scaffold(
            scaffold_descriptor,
            ordered_contig_descriptors,
        )
        out_str += '\n'
        return out_str

    def get_fasta_record_for_single_contig_not_in_scaffold(
            self,
            contig_descriptor: ContigDescriptor,
            contig_direction: ContigDirection,
    ) -> str:
        contig_name = contig_descriptor.contig_name
        out_str = ''
        out_str += f'>{contig_name}\n'
        out_str += self.get_cropped_dna_string_for_single_contig(
            contig_descriptor,
            contig_direction
        )
        out_str += '\n'
        return out_str

    def finalize_fasta_for_assembly(
            self,
            file_like,
            ordered_finalization_records: List[Tuple[Optional[ScaffoldDescriptor], List[Tuple[ContigDescriptor, ContigDirection]]]],
    ):
        for opt_scaffold, ctgs in ordered_finalization_records:
            if opt_scaffold is None:
                assert (
                    len(ctgs) == 1
                ), "Finalization record for single contig not in scaffold must contain only one contig descriptor"
                contig_descriptor, contig_direction = ctgs[0]
                contig_record_str: str = self.get_fasta_record_for_single_contig_not_in_scaffold(
                    contig_descriptor,
                    contig_direction,
                )
                contig_record_bytes: bytes = contig_record_str.encode(
                    encoding='utf-8'
                )
                file_like.write(contig_record_bytes)
            else:
                assert (
                    len(ctgs) > 0
                ), "Finalization record for scaffold must contain at least one contig"
                scaffold_record_str: str = self.get_fasta_record_for_scaffold(
                    opt_scaffold,
                    ctgs,
                )
                scaffold_record_bytes: bytes = scaffold_record_str.encode(
                    encoding='utf-8'
                )
                file_like.write(scaffold_record_bytes)
