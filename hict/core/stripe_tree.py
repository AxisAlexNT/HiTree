# https://neerc.ifmo.ru/wiki/index.php?title=Декартово_дерево
import random
import sys
import threading
from typing import Optional, Tuple, Callable, List, NamedTuple

import numpy as np

from hict.core.common import StripeDescriptor, ContigDirection, ContigHideType

from readerwriterlock import rwlock

from dataclasses import dataclass

# random.seed(706258761087560)


class StripeTree(object):
    resolution: np.int64
    tree_lock: rwlock.RWLockWrite
    cache: 'StripeTreeCache'
    stripe_descriptors_by_id: List[Optional[StripeDescriptor]]

    class ExposedSegment(NamedTuple):
        less: Optional['StripeTree.Node']
        segment: Optional['StripeTree.Node']
        greater: Optional['StripeTree.Node']

    class Node:
        class NodeSize(NamedTuple):
            length_bins: np.int64
            block_count: np.int64

        stripe_descriptor: StripeDescriptor
        subtree_count: np.int64
        subtree_length: np.int64
        containing_tree: 'StripeTree'

        def __init__(self,
                     stripe_descriptor: StripeDescriptor,
                     containing_tree: 'StripeTree'
                     ) -> None:
            super().__init__()
            self.stripe_descriptor = stripe_descriptor
            self.y_priority: np.int64 = np.int64(
                random.randint(1 - sys.maxsize, sys.maxsize - 1))
            self.left: Optional[StripeTree.Node] = None
            self.right: Optional[StripeTree.Node] = None
            self.subtree_length: np.int64 = self.stripe_descriptor.stripe_length_bins
            self.subtree_count: np.int64 = np.int64(
                1)  # Implicit key for the treap
            self.needs_changing_direction = False
            self.parent: Optional[StripeTree.Node] = None
            self.containing_tree = containing_tree

        def update_sizes(self):
            self.subtree_count = 1
            self.subtree_length = self.stripe_descriptor.stripe_length_bins
            if self.left is not None:
                self.subtree_count += self.left.subtree_count
                self.subtree_length += self.left.subtree_length
            if self.right is not None:
                self.subtree_count += self.right.subtree_count
                self.subtree_length += self.right.subtree_length

        def push(self):
            if self.needs_changing_direction:
                (self.left, self.right) = (self.right, self.left)
                if self.left is not None:
                    self.left.needs_changing_direction = not self.left.needs_changing_direction
                if self.right is not None:
                    self.right.needs_changing_direction = not self.right.needs_changing_direction
                self.needs_changing_direction = False

        def true_direction(self) -> ContigDirection:
            self.push()
            return self.stripe_descriptor.contig_descriptor.direction

        def true_hidden(self) -> ContigHideType:
            self.push()
            return self.stripe_descriptor.contig_descriptor.presence_in_resolution[self.containing_tree.resolution]

        def true_stripe_descriptor(self):
            self.push()
            return self.stripe_descriptor

        def get_sizes(self) -> NodeSize:
            self.update_sizes()
            return StripeTree.Node.NodeSize(self.subtree_length, self.subtree_count)

        def reverse_subtree(self):
            self.needs_changing_direction = not self.needs_changing_direction

        def leftmost(self):
            return StripeTree.get_leftmost(self)

        def rightmost(self):
            return StripeTree.get_rightmost(self)

    root: Optional[Node]

    def __init__(self, resolution: np.int64, random_seed: Optional[int] = None) -> None:
        super().__init__()
        if random_seed is not None:
            random.seed(random_seed)
        self.tree_lock = rwlock.RWLockWrite(lock_factory=threading.RLock)
        self.stripe_descriptors_by_id = []
        with self.tree_lock.gen_wlock():
            self.root = None
            self.resolution = resolution
            self.cache = StripeTreeCache(self)
            self.cache.update()

    def create_node(self, stripe_descriptor: StripeDescriptor, update_cache: bool = False) -> Node:
        stripe_id = stripe_descriptor.stripe_id
        with self.tree_lock.gen_wlock():
            with self.cache.cache_lock.gen_wlock(), self.cache.cv_is_valid:
                self.cache.is_valid = False
            if len(self.stripe_descriptors_by_id) <= stripe_id:
                self.stripe_descriptors_by_id.extend(
                    (1 + stripe_id - len(self.stripe_descriptors_by_id)) * [None])
            self.stripe_descriptors_by_id[stripe_id] = stripe_descriptor
            if update_cache:
                with self.cache.cache_lock.gen_wlock():
                    self.cache.update()
            return StripeTree.Node(stripe_descriptor, self)

    def split_node_by_count(self, t: Optional[Node], k: np.int64) -> Tuple[Optional[Node], Optional[Node]]:
        with self.tree_lock.gen_wlock():
            if t is None:
                return None, None
            left_count: np.int64 = t.left.subtree_count if t.left is not None else np.int64(
                0)
            t.push()
            if left_count >= k:
                (t1, t2) = self.split_node_by_count(t.left, k)
                t.left = t2
                t.update_sizes()
                if t1 is not None:
                    t1.parent = None
                return t1, t
            else:
                (t1, t2) = self.split_node_by_count(
                    t.right, k - left_count - 1)
                t.right = t1
                t.update_sizes()
                if t2 is not None:
                    t2.parent = None
                return t, t2

    def split_node_by_length(
            self,
            t: Optional[Node],
            k: np.int64,
            include_equal_to_the_left: bool = False
    ) -> Tuple[
            Optional[Node], Optional[Node]]:
        """
        Splits input tree into (l, r) so that total length of l is smallest possible >= k (if include = False)
        :param include_equal_to_the_left: Whether to include node where point resides to the left (True) or to the right tree (False)
        :param t: An input tree
        :param k: Splitting parameter
        :return: Tree nodes (l, r) so that total length of l is smallest possible >= k (if include = False)
        """
        with self.tree_lock.gen_wlock():
            if t is None:
                return None, None
            if k <= 0:
                return None, t
            t.push()
            left_length: np.int64 = t.left.subtree_length if t.left is not None else np.int64(
                0)
            if k <= left_length:
                (t1, t2) = self.split_node_by_length(
                    t.left, k, include_equal_to_the_left)
                t.left = t2
                t.update_sizes()
                if t1 is not None:
                    t1.parent = None
                return t1, t
            elif left_length < k <= (left_length + t.stripe_descriptor.stripe_length_bins):
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
                (t1, t2) = self.split_node_by_length(t.right, k - left_length - t.stripe_descriptor.stripe_length_bins,
                                                     include_equal_to_the_left)
                t.right = t1
                t.update_sizes()
                if t2 is not None:
                    t2.parent = None
                return t, t2

    @staticmethod
    def merge_nodes(t1: Optional[Node], t2: Optional[Node]) -> Optional[Node]:
        if t1 is None:
            return t2
        if t2 is None:
            return t1
        t1.push()
        t2.push()
        if t1.y_priority > t2.y_priority:
            t1.right = StripeTree.merge_nodes(t1.right, t2)
            t1.update_sizes()
            if t1.left is not None:
                t1.left.parent = t1
            if t1.right is not None:
                t1.right.parent = t1
            return t1
        else:
            t2.left = StripeTree.merge_nodes(t1, t2.left)
            t2.update_sizes()
            if t2.left is not None:
                t2.left.parent = t2
            if t2.right is not None:
                t2.right.parent = t2
            return t2

    def insert_at_position(self, position: np.int64, contig_block: StripeDescriptor):
        with self.tree_lock.gen_wlock():
            new_node: StripeTree.Node = self.create_node(contig_block)
            if self.root is not None:
                (t1, t2) = self.split_node_by_count(self.root, position)
                t1 = StripeTree.merge_nodes(t1, new_node)
                self.root = StripeTree.merge_nodes(t1, t2)
            else:
                self.root = new_node

    @staticmethod
    def traverse_node(t: Optional[Node], f: Callable[[Node], None]):
        if t is None:
            return

        t.push()

        StripeTree.traverse_node(t.left, f)
        f(t)
        StripeTree.traverse_node(t.right, f)

    def traverse(self, f: Callable[[Node], None]):
        StripeTree.traverse_node(self.root, f)

    def get_length_bins(self):
        with self.tree_lock.gen_wlock():
            if self.root is not None:
                self.root.update_sizes()
                return self.root.subtree_length
            return 0

    def get_node_count(self):
        with self.tree_lock.gen_wlock():
            if self.root is not None:
                self.root.update_sizes()
                return self.root.subtree_count
            return 0

    def expose_segment_by_length(self, start_bins: np.int64, end_bins: np.int64) -> ExposedSegment:
        """
        Exposes segment from (start_bins-1) to end_bins (both inclusive).
        """
        with self.tree_lock.gen_wlock():
            with self.cache.cache_lock.gen_wlock(), self.cache.cv_is_valid:
                self.cache.is_valid = False
            (t_le, t_gr) = self.split_node_by_length(
                self.root, end_bins, include_equal_to_the_left=True)
            # TODO: Actually 1+start_bins
            (t_l, t_seg) = self.split_node_by_length(
                t_le, start_bins, include_equal_to_the_left=False)
            if t_seg is not None:
                t_seg.push()
            return StripeTree.ExposedSegment(t_l, t_seg, t_gr)

    def expose_segment_by_count(self, start_order: np.int64, end_order: np.int64) -> ExposedSegment:
        """
        Exposes segment from (start_order) to end_order (both inclusive).
        """
        with self.tree_lock.gen_wlock():
            with self.cache.cache_lock.gen_wlock(), self.cache.cv_is_valid:
                self.cache.is_valid = False
            (t_le, t_gr) = self.split_node_by_count(
                self.root, 1+end_order)
            # TODO: Actually 1+start_bins
            (t_l, t_seg) = self.split_node_by_count(
                t_le, start_order)
            if t_seg is not None:
                t_seg.push()
            return StripeTree.ExposedSegment(t_l, t_seg, t_gr)

    def commit_exposed_segment(self, segm: ExposedSegment, update_cache: bool = True):
        with self.tree_lock.gen_wlock():
            (t_l, t_seg, t_gr) = segm
            t_le = StripeTree.merge_nodes(t_l, t_seg)
            self.root = StripeTree.merge_nodes(t_le, t_gr)
            if update_cache:
                with self.cache.cache_lock.gen_wlock():
                    self.cache.update()

    def reverse_direction_in_bins(self, start_bins: np.int64, end_bins: np.int64, update_cache: bool = True):
        start_order: Optional[int] = None
        end_order: Optional[int] = None
        with self.tree_lock.gen_wlock():
            exposed_segment = self.expose_segment_by_length(
                start_bins, end_bins)
            if exposed_segment.segment is not None:
                exposed_segment.segment.reverse_subtree()
                start_order = exposed_segment.less.get_sizes().block_count
                end_order = exposed_segment.segment.get_sizes().block_count + start_order
            self.commit_exposed_segment(exposed_segment, update_cache=False)
            if update_cache and start_order is not None and end_order is not None:
                with self.cache.cache_lock.gen_wlock():
                    self.cache.update((start_order, end_order))

    @staticmethod
    def find_node_by_length(t: Optional[Node], length_bins: np.int64) -> Optional[Node]:
        if t is None:
            return None
        if length_bins == t.subtree_length:
            return t
        t.push()
        left_subtree_length: np.int64 = (
            t.left.subtree_length if t.left is not None else np.int64(0))
        if length_bins <= left_subtree_length:
            return StripeTree.find_node_by_length(t.left, length_bins)
        elif left_subtree_length < length_bins <= (left_subtree_length + t.stripe_descriptor.stripe_length_bins):
            return t
        elif (left_subtree_length + t.stripe_descriptor.stripe_length_bins) < length_bins:
            return StripeTree.find_node_by_length(
                t.right,
                length_bins - t.stripe_descriptor.stripe_length_bins - left_subtree_length
            )
        else:
            raise Exception("Impossible case??")

    def find_stripe_storing_bins(self, length_bins: np.int64) -> Optional[StripeDescriptor]:
        with self.tree_lock.gen_rlock():
            mb_node: Optional[StripeTree.Node] = StripeTree.find_node_by_length(
                self.root, 1 + length_bins)
            if mb_node is not None:
                return mb_node.stripe_descriptor
            else:
                return None

    @dataclass
    class StripesInRange:
        stripes: List[StripeDescriptor]
        first_stripe_start_bins: np.int64

    def get_stripes_in_segment(self, start_bins: np.int64, end_bins: np.int64) -> StripesInRange:
        with self.tree_lock.gen_rlock():
            with self.cache.cache_lock.gen_rlock():
                return self.cache.get_stripes_in_segment(start_bins, end_bins)
        # with self.tree_lock.gen_wlock():
        #     result: List[StripeDescriptor] = []
        #     es: StripeTree.ExposedSegment = self.expose_segment_by_length(
        #         start_bins, end_bins)
        #     StripeTree.traverse_node(
        #         es.segment, lambda n: result.append(n.true_stripe_descriptor()))
        #     first_stripe_start_bins: np.int64 = (
        #         es.less.get_sizes().length_bins
        #     ) if es.less is not None else np.int64(0)
        #     self.commit_exposed_segment(es)
        #     return StripeTree.StripesInRange(result, first_stripe_start_bins)

    def get_stripes_for_rectangle(self, x0_bins: np.int64, y0_bins: np.int64, x1_bins: np.int64, y1_bins: np.int64) -> \
            Tuple[StripesInRange, StripesInRange]:
        with self.tree_lock.gen_wlock():
            return self.get_stripes_in_segment(x0_bins, x1_bins), self.get_stripes_in_segment(y0_bins, y1_bins)

    def move_stripes(self, start_bins: int, end_bins: int, new_start_position_bins: int, update_cache: bool = True) -> None:
        with self.tree_lock.gen_wlock():
            (mt_less, mt_segment, mt_greater) = self.expose_segment_by_length(
                start_bins,
                end_bins
            )
            if mt_segment is not None:
                start_order = mt_less.get_sizes().block_count if mt_less is not None else 0
                end_order = (mt_segment.get_sizes(
                ).block_count if mt_segment is not None else 0) + start_order
                mt_intermediate = StripeTree.merge_nodes(mt_less, mt_greater)
                (mt_new_less, mt_new_greater) = self.split_node_by_length(
                    mt_intermediate,
                    new_start_position_bins,
                    True
                )
                target_order = mt_new_less.get_sizes().block_count if mt_new_less is not None else 0
                mt_new_less_with_segment = StripeTree.merge_nodes(
                    mt_new_less, mt_segment)
                self.root = StripeTree.merge_nodes(
                    mt_new_less_with_segment, mt_new_greater)
                if update_cache:
                    with self.cache.cache_lock.gen_wlock():
                        self.cache.on_move(
                            start_order, end_order, target_order)
            else:
                self.commit_exposed_segment(StripeTree.ExposedSegment(
                    mt_less, mt_segment, mt_greater), False)
                with self.cache.cache_lock.gen_wlock(), self.cache.cv_is_valid:
                    self.cache.is_valid = True
                    self.cache.cv_is_valid.notify_all()

    @staticmethod
    def get_rightmost(node: Optional['StripeTree.Node']) -> Optional['StripeTree.Node']:
        if node is None:
            return None
        current_node: StripeTree.Node = node
        while current_node.right is not None:
            current_node.push()
            current_node = current_node.right
        return current_node

    @staticmethod
    def get_leftmost(node: Optional['StripeTree.Node']) -> Optional['StripeTree.Node']:
        if node is None:
            return None
        current_node: StripeTree.Node = node
        while current_node.left is not None:
            current_node.push()
            current_node = current_node.left
        return current_node

    @staticmethod
    def pn(n: Node):
        print(
            f"Node with stripe_id={n.stripe_descriptor.stripe_id} length={n.stripe_descriptor.stripe_length_bins} direction={n.true_direction()}")

    @staticmethod
    def ni(n: Node):
        StripeTree.traverse_node(n, StripeTree.pn)


class StripeTreeCache(object):
    stripe_tree: StripeTree
    stripe_order: np.ndarray
    length_bp: np.ndarray
    length_bins: np.ndarray
    # prefix_sum_length_bp: np.ndarray
    prefix_sum_length_bins: np.ndarray
    stripe_count: np.int64
    cache_lock: rwlock.RWLockWrite
    is_valid: bool
    cv_is_valid: threading.Condition

    def __init__(self, stripe_tree: StripeTree) -> None:
        self.stripe_tree = stripe_tree
        self.cache_lock = rwlock.RWLockWrite(lock_factory=threading.RLock)
        self.is_valid = False
        self.cv_is_valid = threading.Condition(lock=threading.RLock())

    def update(self, borders: Optional[Tuple[int, int]] = None) -> None:
        """
        Update cache to reflect new state of tree

        Args:
            borders (Optional[Tuple[int, int]], optional): Left border(inclusive) and right border(exclusive) of changes in terms of stripes' orders. Defaults to None.
        """
        with self.cache_lock.gen_wlock():
            if borders is None:
                self.copy_stripes()
            else:
                self.partial_copy_stripes(borders)
            self.recalculate_prefix_sums(borders)
            with self.cv_is_valid:
                self.is_valid = True
                self.cv_is_valid.notify_all()

    def copy_stripes(self) -> None:
        index: int = 0

        def traverse_fn(node: StripeTree.Node) -> None:
            nonlocal index
            self.stripe_order[index] = (node.stripe_descriptor.stripe_id)
            # self.length_bp[index] = node.stripe_descriptor.stripe_length_bp
            self.length_bins[index] = node.stripe_descriptor.stripe_length_bins
            index += 1

        with self.cache_lock.gen_wlock():
            with self.stripe_tree.tree_lock.gen_rlock():
                self.stripe_count = (
                    self.stripe_tree.root.get_sizes().block_count
                ) if self.stripe_tree.root is not None else 0
                self.stripe_order = np.zeros(
                    shape=self.stripe_count, dtype=np.int64)
                # self.length_bp = np.zeros(
                #     shape=self.stripe_count, dtype=np.int64)
                self.length_bins = np.zeros(
                    shape=self.stripe_count, dtype=np.int64)
                self.stripe_tree.traverse(traverse_fn)

    def recalculate_prefix_sums(self, borders: Optional[Tuple[int, int]]) -> None:
        with self.cache_lock.gen_wlock():
            if borders is None:
                # self.prefix_sum_length_bp = np.cumsum(
                #     self.length_bp,
                #     dtype=np.int64
                # )
                self.prefix_sum_length_bins = np.cumsum(
                    self.length_bins,
                    dtype=np.int64
                )
            else:
                # np.cumsum(
                #     self.length_bp,
                #     dtype=np.int64,
                #     out=self.prefix_sum_length_bp[borders[0]:borders[1]]
                # )
                np.cumsum(
                    self.length_bins[borders[0]:borders[1]],
                    dtype=np.int64,
                    out=self.prefix_sum_length_bins[borders[0]:borders[1]]
                )
                if borders[0] != 0:
                    # self.prefix_sum_length_bp[borders[0]:borders[1]
                    #                           ] += self.prefix_sum_length_bp[borders[0]-1]
                    self.prefix_sum_length_bins[borders[0]:borders[1]
                                                ] += self.prefix_sum_length_bins[borders[0]-1]

    def partial_copy_stripes(self, borders: Optional[Tuple[int, int]] = None) -> None:
        index: int = borders[0]

        def traverse_fn(node: StripeTree.Node) -> None:
            nonlocal index
            self.stripe_order[index] = (node.stripe_descriptor.stripe_id)
            # self.length_bp[index] = node.stripe_descriptor.stripe_length_bp
            self.length_bins[index] = node.stripe_descriptor.stripe_length_bins
            index += 1

        with self.cache_lock.gen_wlock():
            with self.stripe_tree.tree_lock.gen_wlock():
                es: StripeTree.ExposedSegment = self.stripe_tree.expose_segment_by_count(
                    1+borders[0],
                    borders[1]-1
                )
                StripeTree.traverse_node(es.segment, traverse_fn)
                self.stripe_tree.commit_exposed_segment(es)

    def on_reversal(self, from_index_incl: int, to_index_excl: int) -> None:
        with self.cache_lock.gen_wlock():
            self.stripe_order[from_index_incl, to_index_excl] = np.flip(
                self.stripe_order[from_index_incl, to_index_excl])
            with self.cv_is_valid:
                self.is_valid = True
                self.cv_is_valid.notify_all()

    def on_move(self, from_index_incl: int, to_index_excl: int, target_index: int) -> None:
        with self.cache_lock.gen_wlock():
            if target_index < from_index_incl:
                self.stripe_order[target_index, to_index_excl] = np.roll(
                    self.stripe_order[target_index, to_index_excl], shift=target_index-from_index_incl)
            elif target_index >= from_index_incl:
                self.stripe_order[from_index_incl:target_index+(to_index_excl-from_index_incl+1)] = np.roll(
                    self.stripe_order[from_index_incl:target_index +
                                      (to_index_excl-from_index_incl)],
                    shift=-(to_index_excl-from_index_incl)
                )
            elif from_index_incl < target_index < to_index_excl:
                self.stripe_order[from_index_incl:to_index_excl+(target_index-from_index_incl)] = np.roll(
                    self.stripe_order[from_index_incl:to_index_excl +
                                      (target_index-from_index_incl)],
                    shift=to_index_excl-from_index_incl
                )
            with self.cv_is_valid:
                self.is_valid = True
                self.cv_is_valid.notify_all()

    def get_stripes_in_segment(self, start_bins: np.int64, end_bins: np.int64) -> StripeTree.StripesInRange:
        with self.cv_is_valid:
            while not self.cv_is_valid.wait_for(lambda: self.is_valid, timeout=1.0):
                self.update()
            with self.cache_lock.gen_rlock():
                self.cv_is_valid.wait_for(lambda: self.is_valid)
                istart, iend = np.searchsorted(
                    self.prefix_sum_length_bins, [start_bins, end_bins])
                stripe_indices = self.stripe_order[istart:iend+1]
                return StripeTree.StripesInRange(
                    list(
                        map(lambda i: self.stripe_tree.stripe_descriptors_by_id[i], stripe_indices)),
                    self.prefix_sum_length_bins[istart -
                                                1] if istart > 0 else np.int64(0)
                )
