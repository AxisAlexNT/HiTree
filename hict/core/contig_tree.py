import random
import sys
import threading
import multiprocessing
import multiprocessing.managers
from typing import Dict, Optional, Tuple, List, Callable, NamedTuple, Union
from copy import deepcopy

import numpy as np


from hict.core.common import ContigDirection, ContigDescriptor, LocationInAssembly, QueryLengthUnit, ContigHideType

from readerwriterlock import rwlock

np.seterr(all='raise')


# random.seed(324590754802)

def constrain_coordinate(x_bins: Union[np.int64, int], lower: Union[np.int64, int],
                         upper: Union[np.int64, int]) -> np.int64:
    return max(min(x_bins, upper), lower)


class ContigTree:
    class ExposedSegment(NamedTuple):
        less: Optional['ContigTree.Node']
        segment: Optional['ContigTree.Node']
        greater: Optional['ContigTree.Node']

    class Node:
        contig_descriptor: ContigDescriptor
        # subtree_count: np.int64
        # subtree_count_excluding_hidden: np.int64
        # subtree_length_bins: Dict[np.int64, np.int64]
        # subtree_length_px: Dict[np.int64, np.int64]
        y_priority: np.int64
        left: Optional['ContigTree.Node']
        right: Optional['ContigTree.Node']
        subtree_count: np.int64
        # Second implicit key for the treap (length):
        subtree_length_bins: Dict[np.int64, np.int64]
        # Third implicit key for treap:
        subtree_length_px: Dict[np.int64, np.int64]
        needs_changing_direction: bool
        needs_updating_scaffold_id_in_subtree: bool
        # parent: Optional['ContigTree.Node']
        direction: ContigDirection
        # scaffold_id: Optional[np.int64]

        def __init__(
            self,
            contig_descriptor: ContigDescriptor,
            subtree_count: np.int64,
            subtree_length_bins: Dict[np.int64, np.int64],
            subtree_length_px: Dict[np.int64, np.int64],
            y_priority: np.int64,
            left: Optional['ContigTree.Node'],
            right: Optional['ContigTree.Node'],
            needs_changing_direction: bool,
            # needs_updating_scaffold_id_in_subtree: bool,
            # parent: Optional['ContigTree.Node'],
            direction: ContigDirection,
            # scaffold_id: Optional[np.int64]
        ) -> None:
            super().__init__()
            self.contig_descriptor: ContigDescriptor = contig_descriptor
            self.y_priority: np.int64 = y_priority
            self.left: Optional['ContigTree.Node'] = left
            self.right: Optional['ContigTree.Node'] = right
            self.subtree_count: np.int64 = subtree_count
            # Second implicit key for the treap (length):
            self.subtree_length_bins: Dict[np.int64,
                                           np.int64] = subtree_length_bins
            # Third implicit key for treap:
            self.subtree_length_px: Dict[np.int64,
                                         np.int64] = subtree_length_px
            self.needs_changing_direction: bool = needs_changing_direction
            # self.needs_updating_scaffold_id_in_subtree: bool = needs_updating_scaffold_id_in_subtree
            # self.parent: Optional['ContigTree.Node'] = parent
            self.direction = direction
            # self.scaffold_id = scaffold_id

        @staticmethod
        def make_new_node_from_descriptor(
            contig_descriptor: ContigDescriptor,
            direction: ContigDirection,
            # scaffold_id: Optional[np.int64]
        ) -> 'ContigTree.Node':
            subtree_length_px = dict()
            for resolution, presence in contig_descriptor.presence_in_resolution.items():
                subtree_length_px[resolution] = (
                    contig_descriptor.contig_length_at_resolution[resolution]
                    if presence in (ContigHideType.AUTO_SHOWN, ContigHideType.FORCED_SHOWN) else np.int64(0)
                )
            return ContigTree.Node(
                contig_descriptor=contig_descriptor,
                y_priority=np.int64(
                    random.randint(1 - sys.maxsize, sys.maxsize - 1)),
                left=None,
                right=None,
                # parent=None,
                subtree_count=np.int64(1),
                subtree_length_bins=dict(
                    contig_descriptor.contig_length_at_resolution),
                subtree_length_px=subtree_length_px,
                needs_changing_direction=False,
                # needs_updating_scaffold_id_in_subtree=False,
                direction=direction,
                # scaffold_id=scaffold_id
            )

        @staticmethod
        def clone_node(n: 'ContigTree.Node') -> 'ContigTree.Node':
            return ContigTree.Node(
                contig_descriptor=n.contig_descriptor,
                subtree_count=deepcopy(n.subtree_count),
                subtree_length_bins=deepcopy(n.subtree_length_bins),
                subtree_length_px=deepcopy(n.subtree_length_px),
                left=n.left,
                right=n.right,
                # parent=n.parent,
                needs_changing_direction=deepcopy(n.needs_changing_direction),
                # needs_updating_scaffold_id_in_subtree=deepcopy(
                # n.needs_updating_scaffold_id_in_subtree),
                y_priority=np.int64(n.y_priority),
                direction=deepcopy(n.direction),
                # scaffold_id=deepcopy(n.scaffold_id)
            )

        def clone(self) -> 'ContigTree.Node':
            return ContigTree.Node.clone_node(self)

        def update_sizes(self) -> 'ContigTree.Node':
            new_node = self.clone()
            new_node.subtree_count = 1
            new_node.subtree_length_bins: Dict[np.int64, np.int64] = dict(
                new_node.contig_descriptor.contig_length_at_resolution)
            for resolution, present in new_node.contig_descriptor.presence_in_resolution.items():
                new_node.subtree_length_px[resolution] = (
                    new_node.contig_descriptor.contig_length_at_resolution[resolution] if present in (
                        ContigHideType.AUTO_SHOWN, ContigHideType.FORCED_SHOWN
                    ) else 0
                )
            for resolution in new_node.subtree_length_bins.keys():
                if new_node.left is not None:
                    new_node.subtree_length_bins[resolution] += new_node.left.subtree_length_bins[resolution]
                    new_node.subtree_length_px[resolution] += (
                        new_node.left.subtree_length_px[resolution]
                    )
                if new_node.right is not None:
                    new_node.subtree_length_bins[resolution] += new_node.right.subtree_length_bins[resolution]
                    new_node.subtree_length_px[resolution] += (
                        new_node.right.subtree_length_px[resolution]
                    )
            if new_node.left is not None:
                new_node.subtree_count += new_node.left.subtree_count
            if new_node.right is not None:
                new_node.subtree_count += new_node.right.subtree_count

            return new_node

        def push(self) -> 'ContigTree.Node':
            new_node = self.clone()
            if new_node.left is not None:
                new_node.left = new_node.left.clone()
                # new_node.left.parent = self
            if new_node.right is not None:
                new_node.right = new_node.right.clone()
                # new_node.right.parent = self
            if new_node.needs_changing_direction:
                (new_node.left, new_node.right) = (
                    new_node.right, new_node.left)
                if new_node.left is not None:
                    new_node.left = new_node.left.clone()
                    new_node.left.needs_changing_direction = not new_node.left.needs_changing_direction
                if new_node.right is not None:
                    new_node.right = new_node.right.clone()
                    new_node.right.needs_changing_direction = not new_node.right.needs_changing_direction
                new_node.direction = ContigDirection(
                    1 - new_node.direction.value)
                new_node.needs_changing_direction = False
            # if new_node.needs_updating_scaffold_id_in_subtree:
            #     if new_node.left is not None:
            #         new_node.left = new_node.left.clone()
            #         new_node.left.scaffold_id = new_node.scaffold_id
            #         new_node.left.needs_updating_scaffold_id_in_subtree = True
            #     if new_node.right is not None:
            #         new_node.right = new_node.right.clone()
            #         new_node.right.scaffold_id = new_node.scaffold_id
            #         new_node.right.needs_updating_scaffold_id_in_subtree = True
            #     new_node.needs_updating_scaffold_id_in_subtree = False
            return new_node

        def true_direction(self) -> ContigDirection:
            if not self.needs_changing_direction:
                return self.direction
            else:
                return ContigDirection(1 - self.direction.value)

        def get_sizes(self, update_sizes: bool = True):
            node: ContigTree.Node = self.update_sizes() if update_sizes else self
            return node.subtree_length_bins, node.subtree_count, node.subtree_length_px

        # def reverse_subtree(self):
        #     self.needs_changing_direction = not self.needs_changing_direction

        def leftmost(self, push: bool = True):
            return ContigTree.get_leftmost(self, push)

        def rightmost(self, push: bool = True):
            return ContigTree.get_rightmost(self, push)

    root: Optional[Node] = None

    root_lock: rwlock.RWLockWrite

    contig_name_to_id: Dict[str, int] = dict()
    contig_id_to_name: Dict[int, str] = dict()

    resolutions: np.ndarray

    # contig_id_to_node_in_tree: Dict[np.int64, Node] = dict()

    # contig_id_to_location_in_assembly: Dict[np.int64,
    #                                         LocationInAssembly] = dict()
    # trivial_location_in_assembly: LocationInAssembly

    def __init__(
        self,
        resolutions_ndarray: np.ndarray,
        random_seed: Optional[int] = None,
        mp_manager: Optional[multiprocessing.managers.SyncManager] = None
    ) -> None:
        super().__init__()
        if random_seed is not None:
            random.seed(random_seed)
        assert (
            0 not in resolutions_ndarray
        ), "Resolution 1:0 should not be present as it is used internally to store contig length in base pairs"
        self.root = None
        self.contig_name_to_id: Dict[str, int] = dict()
        self.contig_id_to_name: Dict[int, str] = dict()
        self.resolutions: np.ndarray = np.hstack(
            (np.zeros(shape=(1,), dtype=np.int64), resolutions_ndarray))
        self.mp_manager = mp_manager
        if mp_manager is not None:
            lock_factory = mp_manager.RLock
        else:
            lock_factory = threading.RLock
        self.root_lock = rwlock.RWLockWrite(lock_factory=lock_factory)
        # self.trivial_location_in_assembly = LocationInAssembly(
        #     order=np.int64(0),
        #     start_bp=np.int64(0),
        #     start_bins=dict.fromkeys(resolutions_ndarray, np.int64(0)),
        #     start_px=dict.fromkeys(resolutions_ndarray, np.int64(0))
        # )

    def split_node_by_count(self, t: Optional[Node], k: np.int64) -> Tuple[Optional[Node], Optional[Node]]:
        if t is None:
            return None, None
        new_t = t.push()
        left_count: np.int64 = t.left.subtree_count if t.left is not None else 0
        if left_count >= k:
            (t1, t2) = self.split_node_by_count(new_t.left, k)
            new_t.left = t2
            new_t = new_t.update_sizes()
            # if t1 is not None:
            #     t1.parent = None
            # if t2 is not None:
            #     t2.parent = new_t
            return t1, new_t
        else:
            (t1, t2) = self.split_node_by_count(
                new_t.right, k - left_count - 1)
            new_t.right = t1
            new_t = new_t.update_sizes()
            # if t1 is not None:
            #     t1.parent = new_t
            # if t2 is not None:
            #     t2.parent = None
            return new_t, t2

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
            return None, t.push().update_sizes()
        assert resolution in t.subtree_length_bins.keys(), "Unknown resolution"
        new_t = t.push().update_sizes()
        left_length: np.int64 = 0
        if new_t.left is not None:
            new_t.left = new_t.left.push().update_sizes()
            if exclude_hidden_contigs:
                left_length = new_t.left.subtree_length_px[resolution]
            else:
                left_length = new_t.left.subtree_length_bins[resolution]
        if new_t.right is not None:
            new_t.right = new_t.right.push().update_sizes()
        new_t = new_t.update_sizes()
        # if new_t.left is not None:
        #     new_t.left.parent = new_t
        # if new_t.right is not None:
        #     new_t.right.parent = new_t

        if k <= left_length:
            (t1, t2) = self.split_node_by_length_internal(
                resolution,
                new_t.left,
                k,
                include_equal_to_the_left,
                exclude_hidden_contigs
            )
            if t2 is not None:
                t2 = t2.push().update_sizes()
            new_t.left = t2
            new_t = new_t.push().update_sizes()
            if t1 is not None:
                t1 = t1.push().update_sizes()
                # t1.parent = None
            # if new_t.left is not None:
            #     new_t.left.parent = new_t
            return t1, new_t
        else:
            contig_node_length: np.int64 = (
                0 if (exclude_hidden_contigs and (
                    new_t.contig_descriptor.presence_in_resolution[resolution] in
                    (
                        ContigHideType.AUTO_HIDDEN,
                        ContigHideType.FORCED_HIDDEN,
                    )
                )) else new_t.contig_descriptor.contig_length_at_resolution[resolution]
            )
            if left_length < k < (left_length + contig_node_length):
                if include_equal_to_the_left:
                    t2 = new_t.right
                    new_t.right = None
                    new_t = new_t.push().update_sizes()
                    if t2 is not None:
                        t2 = t2.push().update_sizes()
                        # t2.parent = None
                    return new_t, t2
                else:
                    t1 = new_t.left
                    new_t.left = None
                    new_t = new_t.push().update_sizes()
                    if t1 is not None:
                        t1 = t1.push().update_sizes()
                        # t1.parent = None
                    return t1, new_t

            else:
                (t1, t2) = self.split_node_by_length_internal(
                    resolution,
                    new_t.right,
                    k - (left_length + contig_node_length),
                    include_equal_to_the_left,
                    exclude_hidden_contigs
                )
                new_t.right = t1
                new_t = new_t.push().update_sizes()
                if t1 is not None:
                    t1 = t1.push().update_sizes()
                    # t1.parent = new_t
                if t2 is not None:
                    t2 = t2.push().update_sizes()
                    # t2.parent = None
                return new_t, t2

    def merge_nodes(self, t1: Optional[Node], t2: Optional[Node]) -> Optional[Node]:
        if t1 is None:
            return t2.clone()
        if t2 is None:
            return t1.clone()
        new_t1 = t1.push()
        new_t2 = t2.push()
        if new_t1.y_priority > new_t2.y_priority:
            new_t1.right = self.merge_nodes(new_t1.right, new_t2)
            new_t1 = new_t1.update_sizes()
            # if new_t1.left is not None:
            #     new_t1.left.parent = new_t1
            # if new_t1.right is not None:
            #     new_t1.right.parent = new_t1
            return new_t1
        else:
            new_t2.left = self.merge_nodes(new_t1, new_t2.left)
            new_t2 = new_t2.update_sizes()
            # if new_t2.left is not None:
            #     new_t2.left.parent = new_t2
            # if new_t2.right is not None:
            #     new_t2.right.parent = new_t2
            return new_t2

    # def get_left_subsize(
    #         self,
    #         raw_node: Node,
    #         include_border_in_size: bool
    # ) -> Tuple[Dict[np.int64, np.int64], Dict[np.int64, np.int64], np.int64]:
    #     """
    #     By a given *raw_node* computes a number of nodes that are located to the left of it. *raw_node* might reside
    #     somewhere inside the tree and have not updated flags, this method first ascends to the top node (root) of the
    #     tree, saving *raw* path into the list, and then descends back to that node, updating all affected nodes and
    #     recalculating all the sizes.

    #     :param raw_node: A source node, which position is to be computed.
    #     :param include_border_in_size: If True, computes number of nodes that are no more right that a given one (<= in essence). In other words, adds 1 to the final result indicating the given node itself.
    #     :return: A pair of dict mapping resolution --> length and node count
    #     """
    #     assert raw_node is not None, "Cannot find location for None-node"
    #     ascending_sequence: List[ContigTree.Node] = [raw_node]
    #     with self.root_lock.gen_wlock():
    #         while ascending_sequence[-1].parent is not None:
    #             ascending_sequence.append(ascending_sequence[-1].parent)

    #         if ascending_sequence[-1] != self.root:
    #             print("Debugger requested -- ascending led to non-root node")

    #         assert (
    #             ascending_sequence[-1] == self.root
    #         ), "Ascending should be terminated at the root of the tree"

    #         for i, node in enumerate(reversed(ascending_sequence)):
    #             assert (
    #                 node is not None
    #             ), "During descending all nodes should not be None"
    #             node.push()
    #             node.update_sizes()

    #         left_subsize_length: Dict[np.int64, np.int64] = dict().fromkeys(
    #             self.resolutions, np.int64(0))
    #         left_subsize_length_excluding_hidden: Dict[np.int64, np.int64] = dict(
    #         ).fromkeys(self.resolutions, np.int64(0))
    #         left_subsize_count: np.int64 = np.int64(0)

    #         if raw_node.left is not None:
    #             for res in self.resolutions:
    #                 left_subsize_length[res] += raw_node.left.subtree_length_bins[res]
    #                 left_subsize_length_excluding_hidden[res] += raw_node.left.subtree_length_px[res]
    #             left_subsize_count += raw_node.left.subtree_count

    #         if include_border_in_size:
    #             for res in self.resolutions:
    #                 left_subsize_length[res] += raw_node.contig_descriptor.contig_length_at_resolution[res]
    #                 left_subsize_length_excluding_hidden[res] += (
    #                     0 if raw_node.contig_descriptor.presence_in_resolution[res] in (
    #                         ContigHideType.AUTO_HIDDEN, ContigHideType.FORCED_HIDDEN
    #                     ) else currentNode.parent.contig_descriptor.contig_length_at_resolution[res]
    #                 )
    #             left_subsize_count += 1

    #         for i, currentNode in enumerate(ascending_sequence[:-1]):
    #             if currentNode == currentNode.parent.right:
    #                 left_subsize_count += 1
    #                 for resolution in currentNode.parent.subtree_length_bins.keys():
    #                     left_subsize_length[resolution] += currentNode.parent.contig_descriptor.contig_length_at_resolution[
    #                         resolution]
    #                     left_subsize_length_excluding_hidden[resolution] += (
    #                         0 if currentNode.parent.contig_descriptor.presence_in_resolution[resolution] in (
    #                             ContigHideType.AUTO_HIDDEN, ContigHideType.FORCED_HIDDEN
    #                         ) else currentNode.parent.contig_descriptor.contig_length_at_resolution[resolution]
    #                     )
    #                 if currentNode.parent.left is not None:
    #                     left_subsize_count += currentNode.parent.left.subtree_count
    #                     for resolution in currentNode.parent.subtree_length_bins.keys():
    #                         left_subsize_length[resolution] += currentNode.parent.left.subtree_length_bins[resolution]
    #                         left_subsize_length_excluding_hidden[resolution] += (
    #                             currentNode.parent.left.subtree_length_px[resolution]
    #                         )

    #         return left_subsize_length, left_subsize_length_excluding_hidden, left_subsize_count

    # def get_updated_contig_node_by_contig_id(self, contig_id: np.int64) -> 'ContigTree.Node':
    #     raw_node: ContigTree.Node = self.contig_id_to_node_in_tree[contig_id]

    #     # False if left son, True if right son
    #     ascending_sequence: List[bool] = []

    #     asc_node: Optional[ContigTree.Node] = raw_node

    #     with self.root_lock.gen_wlock():
    #         while asc_node is not None and asc_node.parent is not None:
    #             if asc_node is asc_node.parent.left:
    #                 ascending_sequence.append(False)
    #             elif asc_node is asc_node.parent.right:
    #                 ascending_sequence.append(True)
    #             else:
    #                 assert False, "Node is not connected to its parent??"
    #             assert asc_node.parent is not None or asc_node is self.root, "Unlinked node that's not a root of tree??"
    #             asc_node = asc_node.parent

    #         desc_node: Optional[ContigTree.Node] = self.root

    #         assert desc_node is not None or len(
    #             ascending_sequence) == 0, "Root is missing but there is ascending sequence leading to it?"

    #         for asc_dir in reversed(ascending_sequence):
    #             assert desc_node is not None, "Descending leads to nonexistent node??"
    #             flip_flag: bool = desc_node.needs_changing_direction
    #             desc_node.push()
    #             desc_node.update_sizes()
    #             desc_dir: bool = asc_dir ^ flip_flag
    #             if desc_dir is False:
    #                 desc_node = desc_node.left
    #             else:
    #                 desc_node = desc_node.right
    #         if desc_node is not None:
    #             desc_node.push()
    #             # desc_node.update_sizes()
    #         return desc_node

    # def get_contig_location(
    #         self,
    #         contig_id: int
    # ) -> Tuple[
    #     ContigDescriptor,
    #     Dict[np.int64, Tuple[np.int64, np.int64]],
    #     Dict[np.int64, Tuple[np.int64, np.int64]],
    #     np.int64
    # ]:
    #     contig_raw_node: ContigTree.Node = self.contig_id_to_node_in_tree[contig_id]
    #     with self.root_lock.gen_wlock():
    #         left_subsize_length, left_subsize_length_excluding_hidden, left_subsize_count = self.get_left_subsize(
    #             contig_raw_node,
    #             False
    #         )
    #         contig_length: Dict[np.int64,
    #                             np.int64] = contig_raw_node.contig_descriptor.contig_length_at_resolution
    #         location_in_resolutions: Dict[np.int64,
    #                                       Tuple[np.int64, np.int64]] = dict()
    #         location_in_resolutions_excluding_hidden: Dict[np.int64, Tuple[np.int64, np.int64]] = dict(
    #         )
    #         for res in self.resolutions:
    #             location_in_resolutions[res] = (
    #                 left_subsize_length[res],
    #                 left_subsize_length[res] + contig_length[res]
    #             )
    #             location_in_resolutions_excluding_hidden[res] = (
    #                 left_subsize_length_excluding_hidden[res],
    #                 left_subsize_length_excluding_hidden[res] + (
    #                     0 if (
    #                         contig_raw_node.contig_descriptor.presence_in_resolution[res] in
    #                         (
    #                             ContigHideType.AUTO_HIDDEN,
    #                             ContigHideType.FORCED_HIDDEN,
    #                         )
    #                     ) else contig_length[res]
    #                 )
    #             )
    #         return contig_raw_node.contig_descriptor, location_in_resolutions, location_in_resolutions_excluding_hidden, left_subsize_count

    # def get_contig_order(
    #         self,
    #         contig_id: int
    # ) -> Tuple[
    #     ContigDescriptor,
    #     np.int64
    # ]:
    #     contig_raw_node: ContigTree.Node = self.contig_id_to_node_in_tree[contig_id]
    #     with self.root_lock.gen_wlock():
    #         _, _, left_subsize_count = self.get_left_subsize(
    #             contig_raw_node,
    #             False
    #         )
    #         return contig_raw_node.contig_descriptor, left_subsize_count

    def insert_at_position(
        self,
        contig_descriptor: ContigDescriptor,
        index: np.int64,
        direction: ContigDirection,
        # update_tree: bool = True
    ) -> None:
        new_node: ContigTree.Node = ContigTree.Node.make_new_node_from_descriptor(
            contig_descriptor,
            direction=direction,
        )
        with self.root_lock.gen_wlock():
            # self.contig_id_to_node_in_tree[contig_descriptor.contig_id] = new_node
            if self.root is not None:
                (l, r) = self.split_node_by_count(self.root, index)
                new_l: ContigTree.Node = self.merge_nodes(l, new_node)
                self.root = self.merge_nodes(new_l, r)
                # self.root.parent = None
            else:
                self.root = new_node
            # if update_tree:
            #     self.update_tree()

    def get_sizes(self) -> Tuple[Dict[np.int64, np.int64], np.int64, Dict[np.int64, np.int64]]:
        with self.root_lock.gen_rlock():
            if self.root is not None:
                return self.root.update_sizes().get_sizes()
            else:
                return dict({res: 0 for res in self.resolutions}), 0, dict({res: 0 for res in self.resolutions})

    def get_node_count(self):
        with self.root_lock.gen_rlock():
            if self.root is not None:
                self.root.update_sizes()
                return self.root.subtree_count
            return 0

    @staticmethod
    def get_leftmost(
        node: Optional['ContigTree.Node'],
        push: bool = True
    ) -> Optional['ContigTree.Node']:
        if push:
            return ContigTree.get_leftmost_with_push(node)
        else:
            return ContigTree.get_leftmost_no_push(node)

    @staticmethod
    def get_leftmost_with_push(node: Optional['ContigTree.Node']) -> Optional['ContigTree.Node']:
        if node is None:
            return None
        current_node: ContigTree.Node = node.push()
        while current_node.left is not None:
            current_node = current_node.left.push()
        return current_node

    @staticmethod
    def get_leftmost_no_push(node: Optional['ContigTree.Node']) -> Optional['ContigTree.Node']:
        if node is None:
            return None
        current_node: ContigTree.Node = node
        chdir: bool = False
        while True:
            chdir ^= current_node.needs_changing_direction
            next_node = current_node.left if not chdir else current_node.right
            if next_node is None:
                return current_node
            current_node = next_node

    @staticmethod
    def get_rightmost(
        node: Optional['ContigTree.Node'],
        push: bool = True
    ) -> Optional['ContigTree.Node']:
        if push:
            return ContigTree.get_rightmost_with_push(node)
        else:
            return ContigTree.get_rightmost_no_push(node)

    @staticmethod
    def get_rightmost_with_push(node: Optional['ContigTree.Node']) -> Optional['ContigTree.Node']:
        if node is None:
            return None
        current_node: ContigTree.Node = node.push()
        while current_node.right is not None:
            current_node = current_node.right.push()
        return current_node

    @staticmethod
    def get_rightmost_no_push(node: Optional['ContigTree.Node']) -> Optional['ContigTree.Node']:
        if node is None:
            return None
        current_node: ContigTree.Node = node
        chdir: bool = False
        while True:
            chdir ^= current_node.needs_changing_direction
            next_node = current_node.right if not chdir else current_node.left
            if next_node is None:
                return current_node
            current_node = next_node

    def expose_segment_by_count(self, start_count: np.int64, end_count: np.int64) -> ExposedSegment:
        """
        Exposes segment of contigs in assembly order from start_count to end_count (both inclusive).
        """
        with self.root_lock.gen_rlock():
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
        with self.root_lock.gen_rlock():
            (t_le, t_gr) = self.split_node_by_length(resolution,
                                                     self.root, end_px, include_equal_to_the_left=True)
            (t_l, t_seg) = self.split_node_by_length(
                resolution, t_le, start_px, include_equal_to_the_left=False)
            if t_seg is not None:
                t_seg.push()
            return ContigTree.ExposedSegment(t_l, t_seg, t_gr)

    def expose_segment(
            self,
            resolution: np.int64,
            start_incl: np.int64,
            end_excl: np.int64,
            units: QueryLengthUnit,
    ) -> ExposedSegment:
        """
        Exposes segment from start to end units (both inclusive).
        """
        with self.root_lock.gen_rlock():
            total_assembly_length_in_units = (
                self.root.get_sizes()[[0, 0, 2][units.value]][resolution]
            ) if self.root is not None else 0
            (t_le, t_gr) = self.split_node_by_length(
                resolution,
                self.root,
                end_excl,
                include_equal_to_the_left=True,
                units=units
            )
            t_le_size_in_units = t_le.get_sizes(
            )[[0, 0, 2][units.value]][resolution] if t_le is not None else np.int64(0)
            expected_end = constrain_coordinate(
                end_excl, 0, total_assembly_length_in_units)
            assert (
                t_le_size_in_units >= expected_end
            ), f"After splitting less-or-equal segment ends earlier than queried {t_le_size_in_units} >= {expected_end}?? Assembly length is {total_assembly_length_in_units} and end_excl is {end_excl}"
            (t_l, t_seg) = self.split_node_by_length(
                resolution,
                t_le,
                start_incl,
                include_equal_to_the_left=False,
                units=units
            )
            t_l_size_in_units = t_l.get_sizes(
            )[[0, 0, 2][units.value]][resolution] if t_l is not None else np.int64(0)
            assert (
                t_l_size_in_units <= start_incl
            ), "After splitting less segment starts not when queried {t_l_size_in_units} <= {start_incl}?? Assembly length is {total_assembly_length}"
            if t_seg is not None:
                t_seg = t_seg.push().update_sizes()
                if units == QueryLengthUnit.PIXELS:
                    assert (
                        t_seg.get_sizes()[
                            2][resolution] >= expected_end-start_incl
                    ), f"Total segment length in pixels {t_seg.get_sizes()[2][resolution]} is less than queried [start, end] of {expected_end-start_incl}??"
                elif units == QueryLengthUnit.BINS:
                    assert (
                        t_seg.get_sizes()[
                            0][resolution] >= expected_end-start_incl
                    ), f"Total segment length in bins {t_seg.get_sizes()[0][resolution]} is less than queried [start, end] of {expected_end-start_incl}??"
                elif units == QueryLengthUnit.BASE_PAIRS:
                    assert (
                        t_seg.get_sizes()[0][0] >= expected_end-start_incl
                    ), f"Total segment length in bps {t_seg.get_sizes()[0][0]} is less than queried [start, end] of {expected_end-start_incl}??"

            return ContigTree.ExposedSegment(t_l, t_seg, t_gr)

    def commit_exposed_segment(self, segm: ExposedSegment):
        with self.root_lock.gen_wlock():
            (t_l, t_seg, t_gr) = segm
            t_le = self.merge_nodes(t_l, t_seg)
            self.root = self.merge_nodes(t_le, t_gr)

    def reverse_contigs_in_segment(self, start_index: np.int64, end_index: np.int64):
        """
        Reverses contigs between two give indices (both inclusive).
        @param start_index: Start index of contig (inclusive).
        @param end_index: End index of contig (inclusive).
        """
        with self.root_lock.gen_rlock():
            segm: ContigTree.ExposedSegment = self.expose_segment_by_count(
                start_index, end_index)
            (t_l, t_seg, t_gr) = segm
            if t_seg is not None:
                t_seg: ContigTree.Node
                t_seg = t_seg.clone()
                t_seg.needs_changing_direction = True
                t_seg = t_seg.push()
                self.commit_exposed_segment((t_l, t_seg, t_gr))

    @staticmethod
    def traverse_node(
        t: Optional[Node],
        f: Callable[[Node], None],
        # check_parent_links: bool = False
    ) -> None:
        if t is None:
            return
        # if check_parent_links:
        #     assert (t.left is None) or (t.left.parent is
        #                                 t), "Left subtree has no parent link"
        #     assert (t.right is None) or (t.right.parent is
        #                                  t), "Right subtree has no parent link"
        new_t = t.push()
        ContigTree.traverse_node(new_t.left, f)
        f(new_t)
        ContigTree.traverse_node(new_t.right, f)

    # def update_subtree_state(self, t: Optional[Node], delta_position: LocationInAssembly) -> Tuple[Optional[Node], LocationInAssembly]:
    #     if t is None:
    #         return None, delta_position
    #     new_t = t.push()
    #     new_l, current_contig_location = self.update_subtree_state(
    #         new_t.left, delta_position)

    #     self.contig_id_to_location_in_assembly[t.contig_descriptor.contig_id] = current_contig_location
    #     next_contig_location = current_contig_location.shifted_by_contig(
    #         t.contig_descriptor)

    #     new_r, new_r_pos = self.update_subtree_state(
    #         new_t.right, next_contig_location)
    #     new_t.left = new_l
    #     new_t.right = new_r
    #     new_new_t = new_t.update_sizes()
    #     # if new_l is not None:
    #     #     new_l.parent = new_new_t
    #     # if new_r is not None:
    #     #     new_r.parent = new_new_t
    #     self.contig_id_to_node_in_tree[new_new_t.contig_descriptor.contig_id] = new_new_t
    #     return new_new_t, new_r_pos

    # def update_tree(self):
    #     with self.root_lock.gen_wlock():
    #         old_root = self.root
    #         new_root, _ = self.update_subtree_state(
    #             old_root,
    #             self.trivial_location_in_assembly
    #         )
    #         self.root = new_root

    @staticmethod
    def traverse_nodes_at_resolution(
        t: Optional[Node],
        resolution: np.int64,
        exclude_hidden: bool,
        f: Callable[[Node], None],
        push: bool = True,
        # check_links=False
    ) -> None:
        if push:
            ContigTree.traverse_nodes_at_resolution_with_pushes(
                t,
                resolution,
                exclude_hidden,
                f,
                # check_links
            )
        else:
            ContigTree.traverse_nodes_at_resolution_no_push(
                t,
                resolution,
                exclude_hidden,
                f,
                # check_links
            )

    @staticmethod
    def traverse_nodes_at_resolution_no_push(
        t: Optional[Node],
        resolution: np.int64,
        exclude_hidden: bool,
        f: Callable[[Node], None],
        # check_links: bool = False
    ) -> None:
        if t is None:
            return
        # if check_links:
        #     assert (t.left is None) or (t.left.parent is
        #                                 t), "Left subtree has no parent link"
        #     assert (t.right is None) or (t.right.parent is
        #                                  t), "Right subtree has no parent link"
        ContigTree.traverse_nodes_at_resolution_no_push(
            t.left if not t.needs_changing_direction else t.right,
            resolution,
            exclude_hidden,
            f,
            # check_links
        )
        if not exclude_hidden or t.contig_descriptor.presence_in_resolution[resolution] in (
                ContigHideType.AUTO_SHOWN,
                ContigHideType.FORCED_SHOWN
        ):
            f(t)
        ContigTree.traverse_nodes_at_resolution_no_push(
            t.right if not t.needs_changing_direction else t.left,
            resolution,
            exclude_hidden,
            f,
            # check_links
        )

    @staticmethod
    def traverse_nodes_at_resolution_with_pushes(
            t: Optional[Node],
            resolution: np.int64,
            exclude_hidden: bool,
            f: Callable[[Node], None],
            # check_links: bool = False
    ) -> None:
        if t is None:
            return
        # if check_links:
        #     assert (t.left is None) or (t.left.parent is
        #                                 t), "Left subtree has no parent link"
        #     assert (t.right is None) or (t.right.parent is
        #                                  t), "Right subtree has no parent link"
        new_t = t.push()
        ContigTree.traverse_nodes_at_resolution_with_pushes(
            new_t.left,
            resolution,
            exclude_hidden,
            f,
            # check_links
        )
        if not exclude_hidden or new_t.contig_descriptor.presence_in_resolution[resolution] in (
                ContigHideType.AUTO_SHOWN,
                ContigHideType.FORCED_SHOWN
        ):
            f(new_t)
        ContigTree.traverse_nodes_at_resolution_with_pushes(
            new_t.right,
            resolution,
            exclude_hidden,
            f,
            # check_links
        )

    def traverse(self, f: Callable[[Node], None]):
        with self.root_lock.gen_rlock():
            ContigTree.traverse_node(self.root, f)

    def traverse_at_resolution(self, resolution: np.int64, exclude_hidden: bool, f: Callable[[Node], None]):
        with self.root_lock.gen_rlock():
            ContigTree.traverse_nodes_at_resolution(
                self.root, resolution, exclude_hidden, f)

    def get_contig_list(self) -> List[
        Tuple[
            ContigDescriptor,
            ContigDirection,
            # Dict[np.int64, Tuple[np.int64, np.int64]]
        ]
    ]:
        descriptors: List[Tuple[ContigDescriptor, ContigDirection,
                                Dict[np.int64, Tuple[np.int64, np.int64]]]] = []

        # position_bins_at_res: Dict[np.int64, np.int64] = dict()

        def traverse_fn(n: ContigTree.Node) -> None:
            # position_at_resolution: Dict[np.int64,
            #                              Tuple[np.int64, np.int64]] = dict()
            # for res, ctg_len in n.contig_descriptor.contig_length_at_resolution.items():
            #     position_at_resolution[res] = (
            #         position_bins_at_res[res], position_bins_at_res[res] + ctg_len)
            #     position_bins_at_res[res] += ctg_len
            descriptors.append(
                (
                    n.contig_descriptor,
                    n.direction,
                    # position_at_resolution
                )
            )

        self.traverse(traverse_fn)

        return descriptors

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
