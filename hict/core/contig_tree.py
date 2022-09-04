import random
import sys
from typing import Dict, Optional, Tuple, List, Callable, NamedTuple

import numpy as np

from hict.core.common import ContigDirection, ContigDescriptor, QueryLengthUnit, ContigHideType

np.seterr(all='raise')

# random.seed(324590754802)

class ContigTree:
    class ExposedSegment(NamedTuple):
        less: Optional['ContigTree.Node']
        segment: Optional['ContigTree.Node']
        greater: Optional['ContigTree.Node']

    class Node:
        contig_descriptor: ContigDescriptor
        subtree_count: np.int64
        # subtree_count_excluding_hidden: np.int64
        subtree_length_bins: Dict[np.int64, np.int64]
        subtree_length_px: Dict[np.int64, np.int64]

        def __init__(self,
                     contig_descriptor: ContigDescriptor
                     ) -> None:
            super().__init__()
            self.contig_descriptor = contig_descriptor
            self.y_priority: np.int64 = np.int64(random.randint(1 - sys.maxsize, sys.maxsize - 1))
            self.left: Optional[ContigTree.Node] = None
            self.right: Optional[ContigTree.Node] = None
            # First implicit key for the treap (count):
            self.subtree_count: np.int64 = np.int64(1)
            # Second implicit key for the treap (length):
            self.subtree_length_bins = dict(contig_descriptor.contig_length_at_resolution)
            # Third implicit key for treap:
            self.subtree_length_px = dict()
            for resolution, present in contig_descriptor.presence_in_resolution.items():
                self.subtree_length_px[resolution] = (
                    contig_descriptor.contig_length_at_resolution[resolution] if present else np.int64(0)
                )
            self.needs_changing_direction: bool = False
            self.needs_updating_scaffold_id_in_subtree: bool = False
            self.parent: Optional[ContigTree.Node] = None

        def update_sizes(self):
            self.subtree_count = 1
            self.subtree_length_bins: Dict[np.int64, np.int64] = dict(self.contig_descriptor.contig_length_at_resolution)
            for resolution, present in self.contig_descriptor.presence_in_resolution.items():
                self.subtree_length_px[resolution] = (
                    self.contig_descriptor.contig_length_at_resolution[resolution] if present in (
                        ContigHideType.AUTO_SHOWN, ContigHideType.FORCED_SHOWN
                    ) else 0
                )
            for resolution in self.subtree_length_bins.keys():
                if self.left is not None:
                    self.subtree_length_bins[resolution] += self.left.subtree_length_bins[resolution]
                    self.subtree_length_px[resolution] += (
                        self.left.subtree_length_px[resolution]
                    )
                if self.right is not None:
                    self.subtree_length_bins[resolution] += self.right.subtree_length_bins[resolution]
                    self.subtree_length_px[resolution] += (
                        self.right.subtree_length_px[resolution]
                    )
            if self.left is not None:
                self.subtree_count += self.left.subtree_count
            if self.right is not None:
                self.subtree_count += self.right.subtree_count

        def push(self) -> None:
            if self.needs_changing_direction:
                (self.left, self.right) = (self.right, self.left)
                if self.left is not None:
                    self.left.needs_changing_direction = \
                        self.left.needs_changing_direction ^ self.needs_changing_direction
                if self.right is not None:
                    self.right.needs_changing_direction = \
                        self.right.needs_changing_direction ^ self.needs_changing_direction
                self.contig_descriptor.direction = ContigDirection(1 - self.contig_descriptor.direction.value)
                self.needs_changing_direction = False
            if self.needs_updating_scaffold_id_in_subtree:
                if self.left is not None:
                    self.left.contig_descriptor.scaffold_id = self.contig_descriptor.scaffold_id
                    self.left.needs_updating_scaffold_id_in_subtree = True
                if self.right is not None:
                    self.right.contig_descriptor.scaffold_id = self.contig_descriptor.scaffold_id
                    self.right.needs_updating_scaffold_id_in_subtree = True
                self.needs_updating_scaffold_id_in_subtree = False

        def true_direction(self) -> ContigDirection:
            self.push()
            return self.contig_descriptor.direction

        def true_contig_descriptor(self) -> ContigDescriptor:
            self.push()
            return self.contig_descriptor

        def get_sizes(self):
            self.update_sizes()
            return self.subtree_length_bins, self.subtree_count, self.subtree_length_px

        def reverse_subtree(self):
            self.needs_changing_direction = not self.needs_changing_direction

        def leftmost(self):
            ContigTree.get_leftmost(self)

        def rightmost(self):
            ContigTree.get_rightmost(self)

    root: Optional[Node] = None

    contig_name_to_id: Dict[str, int] = dict()
    contig_id_to_name: Dict[int, str] = dict()

    resolutions: np.ndarray

    def __init__(self, resolutions_ndarray: np.ndarray, random_seed: Optional[int] = None) -> None:
        super().__init__()
        if random_seed is not None:
            random.seed(random_seed)
        assert (
                0 not in resolutions_ndarray
        ), "Resolution 1:0 should not be present as it is used internally to store contig length in base pairs"
        self.root = None
        self.contig_name_to_id: Dict[str, int] = dict()
        self.contig_id_to_name: Dict[int, str] = dict()
        self.resolutions: np.ndarray = np.hstack((np.zeros(shape=(1,), dtype=np.int64), resolutions_ndarray))

    contig_id_to_node_in_tree: Dict[np.int64, Node] = dict()

    def split_node_by_count(self, t: Optional[Node], k: np.int64) -> Tuple[Optional[Node], Optional[Node]]:
        if t is None:
            return None, None
        left_count: np.int64 = t.left.subtree_count if t.left is not None else 0
        t.push()
        if left_count >= k:
            (t1, t2) = self.split_node_by_count(t.left, k)
            t.left = t2
            t.update_sizes()
            if t1 is not None:
                t1.parent = None
            return t1, t
        else:
            (t1, t2) = self.split_node_by_count(t.right, k - left_count - 1)
            t.right = t1
            t.update_sizes()
            if t2 is not None:
                t2.parent = None
            return t, t2

    def split_node_by_length(
            self,
            resolution: np.int64,
            t: Optional[Node],
            k: np.int64,
            include_equal_to_the_left: bool = False,
            units: QueryLengthUnit = QueryLengthUnit.BINS
    ) -> Tuple[Optional[Node], Optional[Node]]:
        if units == QueryLengthUnit.BASE_PAIRS:
            assert (
                    resolution == 0 or resolution == 1
            ), "Base pairs have resolution 1:1 and are stored as 1:0"
            return self.split_node_by_length_internal(
                resolution,
                t,
                k,
                include_equal_to_the_left,
                False
            )
        elif units == QueryLengthUnit.BINS:
            assert (
                    resolution != 0 and resolution != 1
            ), "Bins query should use actual resolution, not reserved 1:0 or 1:1"
            return self.split_node_by_length_internal(
                resolution,
                t,
                k,
                include_equal_to_the_left,
                False
            )
        elif units == QueryLengthUnit.PIXELS:
            assert (
                    resolution != 0 and resolution != 1
            ), "Pixels query should use actual resolution, not reserved 1:0 or 1:1"
            return self.split_node_by_length_internal(
                resolution,
                t,
                k,
                include_equal_to_the_left,
                True
            )
        else:
            raise Exception("Unknown length unit")

    def split_node_by_length_internal(
            self,
            resolution: np.int64,
            t: Optional[Node],
            k: np.int64,
            include_equal_to_the_left: bool = False,
            exclude_hidden_contigs: bool = False
    ) -> Tuple[Optional[Node], Optional[Node]]:
        """
`           Splits input tree into (l, r) so that total length of l is smallest possible >= k (if include = False)
        :param resolution: resolution in which pixel count (length) is expressed
        :param include_equal_to_the_left: Whether to include node where point resides to the left (True) or to the right tree (False)
        :param t: An input tree
        :param k: Splitting parameter
        :return: Tree nodes (l, r) so that total length of l is smallest possible >= k (if include = False)
        """
        if t is None:
            return None, None
        if k <= 0:
            return None, t
        assert resolution in t.subtree_length_bins.keys(), "Unknown resolution"
        t.push()
        left_length: np.int64 = (
            (t.left.subtree_length_bins[resolution] if t.left is not None else 0)
            if not exclude_hidden_contigs else
            (t.left.subtree_length_px[resolution] if t.left is not None else 0)
        )
        if k <= left_length:
            (t1, t2) = self.split_node_by_length_internal(
                resolution,
                t.left,
                k,
                include_equal_to_the_left,
                exclude_hidden_contigs
            )
            t.left = t2
            t.update_sizes()
            if t1 is not None:
                t1.parent = None
            return t1, t
        else:
            contig_node_length: np.int64 = (
                0 if (exclude_hidden_contigs and (
                        t.contig_descriptor.presence_in_resolution[resolution] in
                        (
                            ContigHideType.AUTO_HIDDEN,
                            ContigHideType.FORCED_HIDDEN,
                        )
                )) else t.contig_descriptor.contig_length_at_resolution[resolution]
            )
            if left_length < k <= (left_length + contig_node_length):
                if include_equal_to_the_left:
                    t2 = t.right
                    t.right = None
                    t.update_sizes()
                    if t2 is not None:
                        t2.parent = None
                    return t, t2
                else:
                    t1 = t.left
                    t.left = None
                    t.update_sizes()
                    if t1 is not None:
                        t1.parent = None
                    return t1, t

            else:
                (t1, t2) = self.split_node_by_length_internal(
                    resolution,
                    t.right,
                    k - (left_length + contig_node_length),
                    include_equal_to_the_left,
                    exclude_hidden_contigs
                )
            t.right = t1
            t.update_sizes()
            if t2 is not None:
                t2.parent = None
            return t, t2

    def merge_nodes(self, t1: Optional[Node], t2: Optional[Node]) -> Optional[Node]:
        if t1 is None:
            return t2
        if t2 is None:
            return t1
        t1.push()
        t2.push()
        if t1.y_priority > t2.y_priority:
            t1.right = self.merge_nodes(t1.right, t2)
            t1.update_sizes()
            if t1.left is not None:
                t1.left.parent = t1
            if t1.right is not None:
                t1.right.parent = t1
            return t1
        else:
            t2.left = self.merge_nodes(t1, t2.left)
            t2.update_sizes()
            if t2.left is not None:
                t2.left.parent = t2
            if t2.right is not None:
                t2.right.parent = t2
            return t2

    def get_left_subsize(
            self,
            raw_node: Node,
            include_border_in_size: bool
    ) -> Tuple[Dict[np.int64, np.int64], Dict[np.int64, np.int64], np.int64]:
        """
        By a given *raw_node* computes a number of nodes that are located to the left of it. *raw_node* might reside
        somewhere inside the tree and have not updated flags, this method first ascends to the top node (root) of the
        tree, saving *raw* path into the list, and then descends back to that node, updating all affected nodes and
        recalculating all the sizes.

        :param raw_node: A source node, which position is to be computed.
        :param include_border_in_size: If True, computes number of nodes that are no more right that a given one (<= in essence). In other words, adds 1 to the final result indicating the given node itself.
        :return: A pair of dict mapping resolution --> length and node count
        """
        # False if left son, True if right son
        ascending_sequence: List[bool] = []

        asc_node: Optional[ContigTree.Node] = raw_node
        while asc_node is not None and asc_node.parent is not None:
            if asc_node is asc_node.parent.left:
                ascending_sequence.append(False)
            elif asc_node is asc_node.parent.right:
                ascending_sequence.append(True)
            else:
                assert False, "Node is not connected to its parent??"
            assert asc_node.parent is not None or asc_node is self.root, "Unlinked node that's not a root of tree??"
            asc_node = asc_node.parent

        left_subsize_length: Dict[np.int64, np.int64] = dict().fromkeys(self.resolutions, np.int64(0))
        left_subsize_length_excluding_hidden: Dict[np.int64, np.int64] = dict().fromkeys(self.resolutions, np.int64(0))
        left_subsize_count: np.int64 = np.int64(0)
        assert self.root is not None, "Operations on empty tree?"
        desc_node: ContigTree.Node = self.root

        for asc_dir in reversed(ascending_sequence):
            assert desc_node is not None, "Descending leads to nonexistent node??"
            flip_flag: bool = desc_node.needs_changing_direction
            desc_node.push()
            desc_node.update_sizes()
            desc_dir: bool = asc_dir ^ flip_flag
            if desc_dir is False:
                assert desc_node.left is not None, "Descending leads to nonexistent node?? (left case)"
                desc_node = desc_node.left
            else:
                if desc_node.left is not None:
                    for res in self.resolutions:
                        left_subsize_length[res] += desc_node.left.subtree_length_bins[res]
                        left_subsize_length_excluding_hidden[res] += desc_node.left.subtree_length_px[res]
                    left_subsize_count += desc_node.left.subtree_count
                for res in self.resolutions:
                    left_subsize_length[res] += desc_node.contig_descriptor.contig_length_at_resolution[res]
                    left_subsize_length_excluding_hidden[res] += desc_node.subtree_length_px[res]
                left_subsize_count += 1
                assert desc_node.right is not None, "Descending leads to nonexistent node?? (right case)"
                desc_node = desc_node.right
        if desc_node is not None:
            desc_node.push()
        # desc_node.update_sizes() # TODO: MB not needed
        if desc_node.left is not None:
            for res in self.resolutions:
                left_subsize_length[res] += desc_node.left.subtree_length_bins[res]
                left_subsize_length_excluding_hidden[res] += desc_node.left.subtree_length_px[res]
            left_subsize_count += desc_node.left.subtree_count
        if include_border_in_size:
            for res in self.resolutions:
                left_subsize_length[res] += desc_node.contig_descriptor.contig_length_at_resolution[res]
                left_subsize_length_excluding_hidden[res] += desc_node.subtree_length_px[res]
            left_subsize_count += 1
        return left_subsize_length, left_subsize_length_excluding_hidden, left_subsize_count

    def get_updated_contig_node_by_contig_id(self, contig_id: np.int64) -> 'ContigTree.Node':
        raw_node: ContigTree.Node = self.contig_id_to_node_in_tree[contig_id]

        # False if left son, True if right son
        ascending_sequence: List[bool] = []

        asc_node: Optional[ContigTree.Node] = raw_node
        while asc_node is not None and asc_node.parent is not None:
            if asc_node is asc_node.parent.left:
                ascending_sequence.append(False)
            elif asc_node is asc_node.parent.right:
                ascending_sequence.append(True)
            else:
                assert False, "Node is not connected to its parent??"
            assert asc_node.parent is not None or asc_node is self.root, "Unlinked node that's not a root of tree??"
            asc_node = asc_node.parent

        desc_node: Optional[ContigTree.Node] = self.root

        assert desc_node is not None or len(ascending_sequence) == 0, "Root is missing but there is ascending sequence leading to it?"

        for asc_dir in reversed(ascending_sequence):
            assert desc_node is not None, "Descending leads to nonexistent node??"
            flip_flag: bool = desc_node.needs_changing_direction
            desc_node.push()
            desc_node.update_sizes()
            desc_dir: bool = asc_dir ^ flip_flag
            if desc_dir is False:
                desc_node = desc_node.left
            else:
                desc_node = desc_node.right
        if desc_node is not None:
            desc_node.push()
            # desc_node.update_sizes()
        return desc_node

    def get_contig_location(
            self,
            contig_id: int
    ) -> Tuple[
        ContigDescriptor,
        Dict[np.int64, Tuple[np.int64, np.int64]],
        Dict[np.int64, Tuple[np.int64, np.int64]],
        np.int64
    ]:
        contig_raw_node: ContigTree.Node = self.contig_id_to_node_in_tree[contig_id]
        left_subsize_length, left_subsize_length_excluding_hidden, left_subsize_count = self.get_left_subsize(
            contig_raw_node,
            False
        )
        contig_length: Dict[np.int64, np.int64] = contig_raw_node.contig_descriptor.contig_length_at_resolution
        location_in_resolutions: Dict[np.int64, Tuple[np.int64, np.int64]] = dict()
        location_in_resolutions_excluding_hidden: Dict[np.int64, Tuple[np.int64, np.int64]] = dict()
        for res in self.resolutions:
            location_in_resolutions[res] = (
                left_subsize_length[res],
                left_subsize_length[res] + contig_length[res]
            )
            location_in_resolutions_excluding_hidden[res] = (
                left_subsize_length_excluding_hidden[res],
                left_subsize_length_excluding_hidden[res] + (
                    0 if (
                            contig_raw_node.contig_descriptor.presence_in_resolution[res] in
                            (
                                ContigHideType.AUTO_HIDDEN,
                                ContigHideType.FORCED_HIDDEN,
                            )
                    ) else contig_length[res]
                )
            )
        return contig_raw_node.contig_descriptor, location_in_resolutions, location_in_resolutions_excluding_hidden, left_subsize_count

    def get_contig_order(
            self,
            contig_id: int
    ) -> Tuple[
        ContigDescriptor,
        np.int64
    ]:
        contig_raw_node: ContigTree.Node = self.contig_id_to_node_in_tree[contig_id]
        _, _, left_subsize_count = self.get_left_subsize(
            contig_raw_node,
            False
        )
        return contig_raw_node.contig_descriptor, left_subsize_count

    def insert_at_position(self, contig_descriptor: ContigDescriptor, index: np.int64):
        new_node: ContigTree.Node = ContigTree.Node(contig_descriptor)
        self.contig_id_to_node_in_tree[contig_descriptor.contig_id] = new_node
        if self.root is not None:
            (l, r) = self.split_node_by_count(self.root, index)
            new_l: ContigTree.Node = self.merge_nodes(l, new_node)
            self.root = self.merge_nodes(new_l, r)
        else:
            self.root = new_node

    def get_sizes(self) -> Tuple[Dict[np.int64, np.int64], np.int64, Dict[np.int64, np.int64]]:
        if self.root is not None:
            return self.root.get_sizes()
        else:
            return dict({0: 0}), 0, dict({0: 0})

    def get_node_count(self):
        if self.root is not None:
            self.root.update_sizes()
            return self.root.subtree_count
        return 0

    @staticmethod
    def get_rightmost(node: Optional['ContigTree.Node']) -> Optional['ContigTree.Node']:
        if node is None:
            return None
        current_node: ContigTree.Node = node
        while current_node.right is not None:
            current_node.push()
            current_node = current_node.right
        return current_node

    @staticmethod
    def get_leftmost(node: Optional['ContigTree.Node']) -> Optional['ContigTree.Node']:
        if node is None:
            return None
        current_node: ContigTree.Node = node
        while current_node.left is not None:
            current_node.push()
            current_node = current_node.left
        return current_node

    def expose_segment_by_count(self, start_count: np.int64, end_count: np.int64) -> ExposedSegment:
        """
        Exposes segment of contigs in assembly order from start_count to end_count (both inclusive).
        """
        (t_le, t_gr) = self.split_node_by_count(self.root, 1 + end_count)
        (t_l, t_seg) = self.split_node_by_count(t_le, start_count)
        if t_seg is not None:
            t_seg.push()
            t_seg.update_sizes()
        return ContigTree.ExposedSegment(t_l, t_seg, t_gr)

    def expose_segment_by_length(self, start_px: np.int64, end_px: np.int64, resolution: np.int64) -> ExposedSegment:
        """
        Exposes segment from start_px to end_px (both inclusive).
        """
        (t_le, t_gr) = self.split_node_by_length(resolution, self.root, end_px, include_equal_to_the_left=True)
        (t_l, t_seg) = self.split_node_by_length(resolution, t_le, start_px, include_equal_to_the_left=False)
        if t_seg is not None:
            t_seg.push()
        return ContigTree.ExposedSegment(t_l, t_seg, t_gr)

    def expose_segment(
            self,
            resolution: np.int64,
            start: np.int64,
            end: np.int64,
            units: QueryLengthUnit,
    ) -> ExposedSegment:
        """
        Exposes segment from start to end units (both inclusive).
        """
        (t_le, t_gr) = self.split_node_by_length(
            resolution,
            self.root,
            end,
            include_equal_to_the_left=True,
            units=units
        )
        (t_l, t_seg) = self.split_node_by_length(
            resolution,
            t_le,
            start,
            include_equal_to_the_left=False,
            units=units
        )
        if t_seg is not None:
            t_seg.push()
        return ContigTree.ExposedSegment(t_l, t_seg, t_gr)

    def commit_exposed_segment(self, segm: ExposedSegment):
        (t_l, t_seg, t_gr) = segm
        t_le = self.merge_nodes(t_l, t_seg)
        self.root = self.merge_nodes(t_le, t_gr)

    def reverse_contigs_in_segment(self, start_index: np.int64, end_index: np.int64):
        segm: ContigTree.ExposedSegment = self.expose_segment_by_count(start_index, end_index)
        (t_l, t_seg, t_gr) = segm
        if t_seg is not None:
            t_seg: ContigTree.Node
            t_seg.reverse_subtree()
        self.commit_exposed_segment(segm)

    @staticmethod
    def traverse_node(t: Optional[Node], f: Callable[[Node], None]):
        if t is None:
            return
        t.push()
        ContigTree.traverse_node(t.left, f)
        f(t)
        ContigTree.traverse_node(t.right, f)

    @staticmethod
    def traverse_nodes_at_resolution(t: Optional[Node], resolution: np.int64, exclude_hidden: bool, f: Callable[[Node], None]):
        if t is None:
            return
        t.push()
        ContigTree.traverse_node(t.left, f)
        if not exclude_hidden or t.contig_descriptor.presence_in_resolution[resolution] in (
                ContigHideType.AUTO_SHOWN,
                ContigHideType.FORCED_SHOWN
        ):
            f(t)
        ContigTree.traverse_node(t.right, f)

    def traverse(self, f: Callable[[Node], None]):
        ContigTree.traverse_node(self.root, f)

    def traverse_at_resolution(self, resolution: np.int64, exclude_hidden: bool, f: Callable[[Node], None]):
        ContigTree.traverse_nodes_at_resolution(self.root, resolution, exclude_hidden, f)

    # def leftmost(self):
    #     return ContigTree.get_leftmost(self.root)
    #
    # def rightmost(self):
    #     return ContigTree.get_rightmost(self.root)

    # # TODO: Remove debug methods:
    # @staticmethod
    # def pn(n: Node):
    #     print(
    #         f"Node contig_id={n.contig_descriptor.contig_id} direction={n.true_direction()} length={str(n.contig_descriptor.contig_length_at_resolution)}")
    #
    # @staticmethod
    # def ni(n: Node):
    #     ContigTree.traverse_node(n, ContigTree.pn)
