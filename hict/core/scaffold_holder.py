from typing import Dict, Optional, Tuple, Set

import numpy as np

from hict.core.common import ScaffoldBorders, ScaffoldDirection, ScaffoldDescriptor


class ScaffoldHolder(object):
    def __init__(self) -> None:
        super().__init__()
        self.scaffold_table: Dict[np.int64, ScaffoldDescriptor] = dict()
        self.scaffold_names: Dict[str, np.int64] = dict()

    def insert_saved_scaffold__(self, descriptor: ScaffoldDescriptor) -> None:
        """
        Restore saved scaffold descriptor from saved state. This method should not be used outside of library's core.

        :param descriptor: Scaffold descriptor to be inserted in scaffold holder.
        """
        self.scaffold_table[descriptor.scaffold_id] = descriptor
        self.scaffold_names[descriptor.scaffold_name] = descriptor.scaffold_id

    def create_scaffold(
            self,
            scaffold_name: Optional[str] = None,
            scaffold_direction: ScaffoldDirection = ScaffoldDirection.FORWARD,
            spacer_length: Optional[int] = 1000,
            scaffold_borders: Optional[ScaffoldBorders] = None,
    ) -> ScaffoldDescriptor:
        """
        Creates a record for a new empty scaffold in the scaffold table and returns its descriptor.
        :param scaffold_name: Human-readable name of this scaffold. These names must be unique for each scaffold. If no name was provided then it is automatically generated in form 'unnamed_scaffold_'+scaffold_name (unique constraint must apply).
        :param scaffold_direction: Initial direction of scaffold (FORWARD as a default)
        :param spacer_length: Number of N's to put between contigs inside this scaffold when processing FASTA query
        :return: Descriptor of newly created empty scaffold.
        """
        scaffold_id = 1+max([np.int64(0)] + list(self.scaffold_table.keys()))
        if scaffold_name is None:
            scaffold_name = f"unnamed_scaffold_{scaffold_id}"
        if scaffold_name in self.scaffold_names.keys():
            raise KeyError(f"Scaffold name is not unique: {scaffold_name}")
        scaffold_descriptor = ScaffoldDescriptor(
            scaffold_id=scaffold_id,
            scaffold_name=scaffold_name,
            scaffold_borders=scaffold_borders,
            scaffold_direction=scaffold_direction,
            spacer_length=spacer_length if spacer_length is not None else 1000
        )
        self.scaffold_table[np.int64(scaffold_id)] = scaffold_descriptor
        self.scaffold_names[scaffold_name] = np.int64(scaffold_id)
        return scaffold_descriptor

    def reverse_scaffold(
            self,
            scaffold_id: Optional[np.int64]
    ) -> Optional[Tuple[np.int64, np.int64]]:
        """
        Reverses scaffold inside scaffold table. Note: this function does not reverse contigs inside ContigTree or stripes in StripeTree!
        :param scaffold_id: Identifier of scaffold that should be reversed.
        :return: IDs of bordering contigs for the given scaffold. These should be reversed using ContigTree.
        """
        if scaffold_id is None:
            return
        if scaffold_id not in self.scaffold_table.keys():
            raise KeyError(
                f"No scaffold with id={scaffold_id} is present in scaffold table")
        scaffold_descriptor: ScaffoldDescriptor = self.scaffold_table[scaffold_id]

        new_scaffold_direction = ScaffoldDirection(
            1 - scaffold_descriptor.scaffold_direction.value)
        new_scaffold_borders: Optional[ScaffoldBorders]
        if scaffold_descriptor.scaffold_borders is not None:
            start_contig_node, end_contig_node = scaffold_descriptor.scaffold_borders
            new_scaffold_borders = ScaffoldBorders(
                end_contig_node,
                start_contig_node
            )
        else:
            new_scaffold_borders = None
        new_scaffold_descriptor: ScaffoldDescriptor = ScaffoldDescriptor(
            scaffold_descriptor.scaffold_id,
            scaffold_descriptor.scaffold_name,
            new_scaffold_borders,
            new_scaffold_direction,
            scaffold_descriptor.spacer_length
        )
        del self.scaffold_table[scaffold_id]
        self.scaffold_table[new_scaffold_descriptor.scaffold_id] = new_scaffold_descriptor
        if scaffold_descriptor.scaffold_borders is not None:
            return start_contig_node, end_contig_node
        else:
            return None

    def remove_scaffold_by_id(
            self,
            scaffold_id: Optional[np.int64]
    ) -> None:
        if scaffold_id is None:
            return
        try:
            scaffold_name: str = self.scaffold_table[scaffold_id].scaffold_name
            del self.scaffold_table[scaffold_id]
            del self.scaffold_names[scaffold_name]
        except KeyError:
            pass

    def get_scaffold_by_id(
            self,
            scaffold_id: np.int64,
    ) -> ScaffoldDescriptor:
        if scaffold_id not in self.scaffold_table.keys():
            raise KeyError(
                f"No scaffold with ID={scaffold_id} is present in scaffold table")
        return self.scaffold_table[scaffold_id]

    def get_scaffold_by_name(
            self,
            scaffold_name: Optional[str],
    ) -> ScaffoldDescriptor:
        if scaffold_name is None or scaffold_name not in self.scaffold_names.keys():
            raise KeyError(
                f"No scaffold named {scaffold_name} is present in scaffold table")
        return self.scaffold_table[self.scaffold_names[scaffold_name]]

    def get_scaffold_id_by_name(
            self,
            scaffold_name: str,
    ) -> np.int64:
        return self.get_scaffold_by_name(scaffold_name).scaffold_id

    def get_scaffold_name_by_id(
            self,
            scaffold_id: np.int64,
    ) -> str:
        return self.get_scaffold_by_id(scaffold_id).scaffold_name

    def remove_unused_scaffolds(self, used_scaffold_ids: Set[np.int64]):
        stored_scaffold_ids: Set[np.int64] = set(self.scaffold_table.keys())
        unused_scaffold_ids: Set[np.int64] = stored_scaffold_ids - \
            used_scaffold_ids
        for unused_scaffold_id in unused_scaffold_ids:
            self.remove_scaffold_by_id(unused_scaffold_id)
        assert (
            len(used_scaffold_ids - stored_scaffold_ids) == 0
        ), "There are scaffold ids in use that are not stored in scaffold_holder??"

    def clear(self) -> None:
        self.scaffold_names.clear()
        self.scaffold_table.clear()
