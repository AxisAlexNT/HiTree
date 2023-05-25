import random
import sys
import threading
import multiprocessing
import multiprocessing.managers
from typing import Dict, Optional, Tuple, List, Callable, NamedTuple, Union
from copy import deepcopy

import numpy as np


from hict.core.common import ContigDirection, ContigDescriptor, QueryLengthUnit, ContigHideType

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
        direction: ContigDirection

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
            direction: ContigDirection,
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
            self.direction = direction

        @staticmethod
        def make_new_node_from_descriptor(
            contig_descriptor: ContigDescriptor,
            direction: ContigDirection,
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
                subtree_count=np.int64(1),
                subtree_length_bins=dict(
                    contig_descriptor.contig_length_at_resolution),
                subtree_length_px=subtree_length_px,
                needs_changing_direction=False,
                direction=direction,
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
                needs_changing_direction=deepcopy(n.needs_changing_direction),
                y_priority=np.int64(n.y_priority),
                direction=deepcopy(n.direction),
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
            if new_node.right is not None:
                new_node.right = new_node.right.clone()
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
            return new_node

        def true_direction(self) -> ContigDirection:
            if not self.needs_changing_direction:
                return self.direction
            else:
                return ContigDirection(1 - self.direction.value)

        def get_sizes(self, update_sizes: bool = True):
            node: ContigTree.Node = self.update_sizes() if update_sizes else self
            return node.subtree_length_bins, node.subtree_count, node.subtree_length_px

        def leftmost(self, push: bool = True):
            return ContigTree.get_leftmost(self, push)

        def rightmost(self, push: bool = True):
            return ContigTree.get_rightmost(self, push)
        
        @staticmethod
        def merge_nodes(t1: Optional['ContigTree.Node'], t2: Optional['ContigTree.Node']) -> Optional['ContigTree.Node']:
            if t1 is None:
                return t2
            if t2 is None:
                return t1
            new_t1 = t1.push()
            new_t2 = t2.push()
            if new_t1.y_priority > new_t2.y_priority:
                new_t1.right = ContigTree.Node.merge_nodes(new_t1.right, new_t2)
                new_t1 = new_t1.update_sizes()
                return new_t1
            else:
                new_t2.left = ContigTree.Node.merge_nodes(new_t1, new_t2.left)
                new_t2 = new_t2.update_sizes()
                return new_t2

    root: Optional[Node] = None

    root_lock: rwlock.RWLockWrite

    contig_name_to_id: Dict[str, int] = dict()
    contig_id_to_name: Dict[int, str] = dict()

    resolutions: np.ndarray

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


    def split_node_by_count(self, t: Optional[Node], k: np.int64) -> Tuple[Optional[Node], Optional[Node]]:
        if t is None:
            return None, None
        new_t = t.push()
        left_count: np.int64 = t.left.subtree_count if t.left is not None else 0
        if left_count >= k:
            (t1, t2) = self.split_node_by_count(new_t.left, k)
            new_t.left = t2
            new_t = new_t.update_sizes()
            return t1, new_t
        else:
            (t1, t2) = self.split_node_by_count(
                new_t.right, k - left_count - 1)
            new_t.right = t1
            new_t = new_t.update_sizes()
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
                    return new_t, t2
                else:
                    t1 = new_t.left
                    new_t.left = None
                    new_t = new_t.push().update_sizes()
                    if t1 is not None:
                        t1 = t1.push().update_sizes()
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
                if t2 is not None:
                    t2 = t2.push().update_sizes()
                return new_t, t2

    def merge_nodes(self, t1: Optional[Node], t2: Optional[Node]) -> Optional[Node]:
        return ContigTree.Node.merge_nodes(t1, t2)

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
    ) -> None:
        if t is None:
            return
        new_t = t.push()
        ContigTree.traverse_node(new_t.left, f)
        f(new_t)
        ContigTree.traverse_node(new_t.right, f)

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
    ) -> None:
        if t is None:
            return
        ContigTree.traverse_nodes_at_resolution_no_push(
            t.left if not t.needs_changing_direction else t.right,
            resolution,
            exclude_hidden,
            f,
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
        )

    @staticmethod
    def traverse_nodes_at_resolution_with_pushes(
            t: Optional[Node],
            resolution: np.int64,
            exclude_hidden: bool,
            f: Callable[[Node], None],
    ) -> None:
        if t is None:
            return
        new_t = t.push()
        ContigTree.traverse_nodes_at_resolution_with_pushes(
            new_t.left,
            resolution,
            exclude_hidden,
            f,
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
        ]
    ]:
        descriptors: List[Tuple[ContigDescriptor, ContigDirection]] = []


        def traverse_fn(n: ContigTree.Node) -> None:
            descriptors.append(
                (
                    n.contig_descriptor,
                    n.direction,
                )
            )

        self.traverse(traverse_fn)

        return descriptors
