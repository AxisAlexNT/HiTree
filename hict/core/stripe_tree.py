# https://neerc.ifmo.ru/wiki/index.php?title=Декартово_дерево
import random
import sys
from typing import Optional, Tuple, Callable, List, NamedTuple

import numpy as np

from hict.core.common import StripeDescriptor, ContigDirection, ContigHideType

# random.seed(706258761087560)


class StripeTree(object):
    resolution: np.int64

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
            self.y_priority: np.int64 = np.int64(random.randint(1 - sys.maxsize, sys.maxsize - 1))
            self.left: Optional[StripeTree.Node] = None
            self.right: Optional[StripeTree.Node] = None
            self.subtree_length: np.int64 = self.stripe_descriptor.stripe_length_bins
            self.subtree_count: np.int64 = np.int64(1)  # Implicit key for the treap
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
        self.root = None
        self.resolution = resolution

    def create_node(self, stripe_descriptor: StripeDescriptor) -> Node:
        return StripeTree.Node(stripe_descriptor, self)

    def split_node_by_count(self, t: Optional[Node], k: np.int64) -> Tuple[Optional[Node], Optional[Node]]:
        if t is None:
            return None, None
        left_count: np.int64 = t.left.subtree_count if t.left is not None else np.int64(0)
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
            t: Optional[Node],
            k: np.int64,
            include_equal_to_the_left: bool = False
    ) -> Tuple[
        Optional[Node], Optional[Node]]:
        """
`           Splits input tree into (l, r) so that total length of l is smallest possible >= k (if include = False)
        :param include_equal_to_the_left: Whether to include node where point resides to the left (True) or to the right tree (False)
        :param t: An input tree
        :param k: Splitting parameter
        :return: Tree nodes (l, r) so that total length of l is smallest possible >= k (if include = False)
        """
        if t is None:
            return None, None
        if k <= 0:
            return None, t
        t.push()
        left_length: np.int64 = t.left.subtree_length if t.left is not None else np.int64(0)
        if k <= left_length:
            (t1, t2) = self.split_node_by_length(t.left, k, include_equal_to_the_left)
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

    def insert_at_position(self, position: np.int64, contig_block: StripeDescriptor):
        new_node: StripeTree.Node = self.create_node(contig_block)
        if self.root is not None:
            (t1, t2) = self.split_node_by_count(self.root, position)
            t1 = self.merge_nodes(t1, new_node)
            self.root = self.merge_nodes(t1, t2)
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
        if self.root is not None:
            self.root.update_sizes()
            return self.root.subtree_length
        return 0

    def get_node_count(self):
        if self.root is not None:
            self.root.update_sizes()
            return self.root.subtree_count
        return 0

    def expose_segment(self, start_bins: np.int64, end_bins: np.int64) -> ExposedSegment:
        """
        Exposes segment from (start_bins-1) to end_bins (both inclusive).
        """
        (t_le, t_gr) = self.split_node_by_length(self.root, end_bins, include_equal_to_the_left=True)
        # TODO: Actually 1+start_bins
        (t_l, t_seg) = self.split_node_by_length(t_le, start_bins, include_equal_to_the_left=False)
        if t_seg is not None:
            t_seg.push()
        return StripeTree.ExposedSegment(t_l, t_seg, t_gr)

    def commit_exposed_segment(self, segm: ExposedSegment):
        (t_l, t_seg, t_gr) = segm
        t_le = self.merge_nodes(t_l, t_seg)
        self.root = self.merge_nodes(t_le, t_gr)

    def reverse_direction_in_bins(self, start_bins: np.int64, end_bins: np.int64):
        exposed_segment = self.expose_segment(start_bins, end_bins)
        if exposed_segment.segment is not None:
            exposed_segment.segment.reverse_subtree()
        self.commit_exposed_segment(exposed_segment)

    @staticmethod
    def find_node_by_length(t: Optional[Node], length_bins: np.int64) -> Optional[Node]:
        if t is None:
            return None
        if length_bins == t.subtree_length:
            return t
        t.push()
        left_subtree_length: np.int64 = (t.left.subtree_length if t.left is not None else np.int64(0))
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

    def find_block_storing_bins(self, length_bins: np.int64) -> Optional[StripeDescriptor]:
        mb_node: Optional[StripeTree.Node] = StripeTree.find_node_by_length(self.root, 1 + length_bins)
        if mb_node is not None:
            return mb_node.stripe_descriptor
        else:
            return None

    def get_blocks_info_in_segment(self, start_bins: np.int64, end_bins: np.int64) -> List[StripeDescriptor]:
        result: List[StripeDescriptor] = []
        es: StripeTree.ExposedSegment = self.expose_segment(start_bins, end_bins)
        StripeTree.traverse_node(es.segment, lambda n: result.append(n.true_stripe_descriptor()))
        self.commit_exposed_segment(es)
        return result

    def get_blocks_for_rectangle(self, x0_bins: np.int64, y0_bins: np.int64, x1_bins: np.int64, y1_bins: np.int64) -> \
            Tuple[List[StripeDescriptor], List[StripeDescriptor]]:
        return self.get_blocks_info_in_segment(x0_bins, x1_bins), self.get_blocks_info_in_segment(y0_bins, y1_bins)

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
