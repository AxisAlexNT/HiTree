import multiprocessing
import multiprocessing.managers
import threading
import random
import sys
from typing import Callable, List, NamedTuple, Optional, Tuple
from hict.core.common import ScaffoldDescriptor
from readerwriterlock import rwlock
import numpy as np


class ScaffoldTree(object):
    class ExposedSegment(NamedTuple):
        less: Optional['ScaffoldTree.Node']
        segment: Optional['ScaffoldTree.Node']
        greater: Optional['ScaffoldTree.Node']

    class Node:
        length_bp: np.int64
        subtree_length_bp: np.int64
        # When scaffold_descriptor is None, node corresponds
        # to the contigs not in scaffold
        scaffold_descriptor: Optional[ScaffoldDescriptor]
        y_priority: np.int64
        left: Optional['ScaffoldTree.Node']
        right: Optional['ScaffoldTree.Node']
        needs_changing_direction: bool

        def __init__(
            self,
            length_bp: np.int64,
            scaffold_descriptor: Optional[ScaffoldDescriptor],
            y_priority: np.int64,
            left: Optional['ScaffoldTree.Node'],
            right: Optional['ScaffoldTree.Node'],
            needs_changing_direction: bool
        ):
            self.length_bp = length_bp
            self.scaffold_descriptor = scaffold_descriptor
            self.y_priority = y_priority
            self.left = left
            self.right = right
            self.needs_changing_direction = needs_changing_direction
            self.subtree_length_bp = length_bp
            if self.left is not None:
                self.subtree_length_bp += self.left.subtree_length_bp
            if self.right is not None:
                self.subtree_length_bp += self.right.subtree_length_bp

        @staticmethod
        def clone_node(node: 'ScaffoldTree.Node') -> 'ScaffoldTree.Node':
            return ScaffoldTree.Node(
                node.length_bp,
                node.scaffold_descriptor,
                node.y_priority,
                node.left,
                node.right,
                node.needs_changing_direction
            )

        def clone(self) -> 'ScaffoldTree.Node':
            return ScaffoldTree.Node.clone_node(self)

        def push(self) -> 'ScaffoldTree.Node':
            new_node = self.clone()
            if new_node.needs_changing_direction:
                (new_node.left, new_node.right) = (
                    new_node.right,
                    new_node.left
                )
            return new_node

        def update_sizes(self) -> 'ScaffoldTree.Node':
            new_node = self.clone()
            new_node.subtree_length_bp = new_node.length_bp
            if new_node.left is not None:
                new_node.subtree_length_bp += new_node.left.subtree_length_bp
            if new_node.right is not None:
                new_node.subtree_length_bp += new_node.right.subtree_length_bp
            return new_node

        @staticmethod
        def split_bp(
            t: Optional['ScaffoldTree.Node'],
            left_size: np.int64,
            include_equal_to_the_left: bool
        ) -> Tuple[Optional['ScaffoldTree.Node'], Optional['ScaffoldTree.Node']]:
            if t is None:
                return None, None
            new_t = t.push()
            left_subtree_length = (
                new_t.left.subtree_length_bp
                if new_t.left is not None else np.int64(0)
            )
            if left_size <= left_subtree_length:
                (t1, t2) = ScaffoldTree.Node.split_bp(
                    new_t.left,
                    left_size,
                    include_equal_to_the_left
                )
                new_t.left = t2
                return t1, new_t.update_sizes()
            elif left_subtree_length < left_size < left_subtree_length + new_t.length_bp:
                if new_t.scaffold_descriptor is not None:
                    if include_equal_to_the_left:
                        t2 = new_t.right
                        new_t.right = None
                        return new_t.update_sizes(), t2
                    else:
                        t1 = new_t.left
                        new_t.left = None
                        return t1, new_t.update_sizes()
                else:
                    t1 = new_t.clone()
                    t1.length_bp = left_size - left_subtree_length
                    t1.right = None

                    right_part_length_bp = new_t.length_bp - t1.length_bp
                    if right_part_length_bp > 0:
                        t2 = new_t.clone()
                        t2.length_bp = right_part_length_bp
                        t2.left = None
                    else:
                        t2 = new_t.right
                    assert t1 is not None
                    assert t2 is not None
                    return t1.update_sizes(), t2.update_sizes()
            else:
                (t1, t2) = ScaffoldTree.Node.split_bp(
                    new_t.right,
                    left_size - left_subtree_length - new_t.length_bp,
                    include_equal_to_the_left
                )
                new_t.right = t1
                return new_t.update_sizes(), t2
            
        @staticmethod
        def leftmost(
            node: Optional['ScaffoldTree.Node']
        ) -> Optional['ScaffoldTree.Node']:
            if node is None:
                return None
            (left, right) = (node.left, node.right)
            if node.needs_changing_direction:
                (left, right) = (right, left)
            if left is None:
                return node
            return ScaffoldTree.Node.leftmost(left)
        
        @staticmethod
        def rightmost(
            node: Optional['ScaffoldTree.Node']
        ) -> Optional['ScaffoldTree.Node']:
            if node is None:
                return None
            (left, right) = (node.left, node.right)
            if node.needs_changing_direction:
                (left, right) = (right, left)
            if right is None:
                return node
            return ScaffoldTree.Node.rightmost(right)

        @staticmethod
        def merge(
            t1: Optional['ScaffoldTree.Node'],
            t2: Optional['ScaffoldTree.Node']
        ) -> Optional['ScaffoldTree.Node']:
            if t1 is None:
                return t2
            if t2 is None:
                return t1
            old_t1_length = t1.subtree_length_bp
            old_t2_length = t2.subtree_length_bp
            t1 = t1.push()
            t2 = t2.push()

            if t1.y_priority > t2.y_priority:
                t1.right = ScaffoldTree.Node.merge(t1.right, t2)
                new_t = ScaffoldTree.Node._optimize_empty_space(t1.update_sizes())
                assert new_t is not None
                assert (
                    new_t.subtree_length_bp == (old_t1_length + old_t2_length)
                ), "Assembly length has changed after merge??"
                return new_t
            else:
                t2.left = ScaffoldTree.Node.merge(t1, t2.left)
                new_t = ScaffoldTree.Node._optimize_empty_space(t2.update_sizes())
                assert new_t is not None
                assert (
                    new_t.subtree_length_bp == (old_t1_length + old_t2_length)
                ), "Assembly length has changed after merge??"
                return new_t

        @staticmethod
        def _optimize_empty_space(
            t: Optional['ScaffoldTree.Node']
        ) -> Optional['ScaffoldTree.Node']:
            if t is None or t.scaffold_descriptor is not None:
                return t

            if t.left is not None and t.left.scaffold_descriptor is None:
                son = t.left
                if son.left is None:
                    t.length_bp += son.length_bp
                    t.left = son.right
                elif son.right is None:
                    t.length_bp += son.length_bp
                    t.left = son.left

            if t.right is not None and t.right.scaffold_descriptor is None:
                son = t.right
                if son.left is None:
                    t.length_bp += son.length_bp
                    t.left = son.right
                elif son.right is None:
                    t.length_bp += son.length_bp
                    t.left = son.left

            return t.update_sizes()

        @staticmethod
        def expose(
            t: Optional['ScaffoldTree.Node'],
            from_bp: np.int64,
            to_bp: np.int64
        ) -> 'ScaffoldTree.ExposedSegment':
            total_length = t.subtree_length_bp if t is not None else np.int64(0)
            to_bp = min(to_bp, total_length)
            from_bp = max(np.int64(0), from_bp)
            le, gr = ScaffoldTree.Node.split_bp(
                t, to_bp, include_equal_to_the_left=True)
            le_size = (le.subtree_length_bp if le is not None else np.int64(0))
            assert (
                le_size == to_bp
            ), "Less-or-equal part does not end where desired??"
            ls, sg = ScaffoldTree.Node.split_bp(
                le, from_bp, include_equal_to_the_left=True)
            less_size = (ls.subtree_length_bp if ls is not None else np.int64(0))
            segment_size = (sg.subtree_length_bp if sg is not None else np.int64(0))
            greater_size = (gr.subtree_length_bp if gr is not None else np.int64(0))
            
            # assert (
            #     less_size == from_bp
            # ), "Less size is not as requested??"
            
            # assert (
            #     (less_size + segment_size) == to_bp
            # ), "Less+Segment do not end as desired??"
            
            assert (
                (less_size + segment_size + greater_size) == total_length
            ), "Exposed segments do not sum up to the total length??"
            
            return ScaffoldTree.ExposedSegment(
                ls,
                sg,
                gr
            )

        @staticmethod
        def traverse(node: Optional['ScaffoldTree.Node'], fun: Callable[['ScaffoldTree.Node'], None]) -> None:
            if node is not None:
                left, right = node.left, node.right
                if node.needs_changing_direction:
                    (left, right) = (right, left)
                if left is not None:
                    ScaffoldTree.Node.traverse(left, fun)
                fun(node)
                if right is not None:
                    ScaffoldTree.Node.traverse(right, fun)

    root: Node
    root_lock: rwlock.RWLockWrite

    def __init__(
        self,
        assembly_length_bp: np.int64,
        mp_manager: Optional[multiprocessing.managers.SyncManager] = None
    ):
        self.root = ScaffoldTree.Node(
            length_bp=assembly_length_bp,
            scaffold_descriptor=None,
            y_priority=np.int64(
                random.randint(
                    1 - sys.maxsize,
                    sys.maxsize - 1
                )
            ),
            left=None,
            right=None,
            needs_changing_direction=False
        )
        if mp_manager is not None:
            lock_factory = mp_manager.RLock
        else:
            lock_factory = threading.RLock
        self.root_lock = rwlock.RWLockWrite(lock_factory=lock_factory)

    def commit_root(
        self,
        exposed_segment: ExposedSegment
    ) -> None:
        le = ScaffoldTree.Node.merge(
            exposed_segment.less, exposed_segment.segment)
        rt = ScaffoldTree.Node.merge(le, exposed_segment.greater)
        assert (
            rt is not None), "Scaffold Tree root must not be none, at least an empty space"
        with self.root_lock.gen_wlock():
            self.root = rt

    def add_scaffold(
        self,
        start_bp_incl: np.int64,
        end_bp_excl: np.int64,
        scaffold_descriptor: ScaffoldDescriptor
    ) -> None:
        with self.root_lock.gen_wlock():
            old_assembly_length_bp = self.root.subtree_length_bp
            es = ScaffoldTree.Node.expose(
                self.root, start_bp_incl, end_bp_excl)
            descriptors: List[ScaffoldDescriptor] = []

            def collect_scaffold(node: ScaffoldTree.Node) -> None:
                if node.scaffold_descriptor is not None:
                    descriptors.append(node.scaffold_descriptor)
            ScaffoldTree.Node.traverse(es.segment, collect_scaffold)
            assert (
                descriptors == []
            ), "Cannot add scaffold into already occupied state"
            self.commit_root(ScaffoldTree.ExposedSegment(
                es.less,
                ScaffoldTree.Node(
                    length_bp=end_bp_excl-start_bp_incl,
                    scaffold_descriptor=scaffold_descriptor,
                    y_priority=np.int64(
                        random.randint(
                            1 - sys.maxsize,
                            sys.maxsize - 1
                        )
                    ),
                    left=None,
                    right=None,
                    needs_changing_direction=False
                ),
                es.greater
            ))
            new_assembly_length_bp = self.root.subtree_length_bp
            assert (
                new_assembly_length_bp == old_assembly_length_bp
            ), "Assembly length has changed after scaffolding??"

    def traverse(self, fun: Callable[[Node], None]) -> None:
        with self.root_lock.gen_rlock():
            ScaffoldTree.Node.traverse(self.root, fun)
