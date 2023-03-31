import multiprocessing
import multiprocessing.managers
import threading
import random
import sys
from typing import Callable, List, NamedTuple, Optional, Tuple
from hict.core.common import ScaffoldDescriptor, ScaffoldBordersBP
from readerwriterlock import rwlock
import numpy as np
import datetime


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
        subtree_scaffolds_count: np.int64

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
            self.subtree_scaffolds_count = (
                np.int64(0) if self.scaffold_descriptor is None else np.int64(1))
            if self.left is not None:
                self.subtree_length_bp += self.left.subtree_length_bp
                self.subtree_scaffolds_count += self.left.subtree_scaffolds_count
            if self.right is not None:
                self.subtree_length_bp += self.right.subtree_length_bp
                self.subtree_scaffolds_count += self.right.subtree_scaffolds_count

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
            new_node.subtree_scaffolds_count = (
                np.int64(0) if new_node.scaffold_descriptor is None else np.int64(1))
            if new_node.left is not None:
                new_node.subtree_length_bp += new_node.left.subtree_length_bp
                new_node.subtree_scaffolds_count += new_node.left.subtree_scaffolds_count
            if new_node.right is not None:
                new_node.subtree_length_bp += new_node.right.subtree_length_bp
                new_node.subtree_scaffolds_count += new_node.right.subtree_scaffolds_count
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
                        t2 = new_t.right.clone() if new_t.right is not None else None
                        new_t.right = None
                        return new_t.update_sizes(), t2
                    else:
                        t1 = new_t.left.clone() if new_t.left is not None else None
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
                        t2.y_priority = np.int64(
                            random.randint(
                                t2.y_priority,
                                sys.maxsize - 1
                            )
                        )
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
                new_t = ScaffoldTree.Node._optimize_empty_space(
                    t1.update_sizes())
                assert new_t is not None
                assert (
                    new_t.subtree_length_bp == (old_t1_length + old_t2_length)
                ), "Assembly length has changed after merge??"
                return new_t
            else:
                t2.left = ScaffoldTree.Node.merge(t1, t2.left)
                new_t = ScaffoldTree.Node._optimize_empty_space(
                    t2.update_sizes())
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
            
            old_subtree_length = t.subtree_length_bp

            if t.left is not None and t.left.scaffold_descriptor is None:
                son = t.left
                if son.right is None:
                    new_t = t.clone()
                    new_t.length_bp += son.length_bp
                    new_t.left = son.left
                    t = new_t

            if t.right is not None and t.right.scaffold_descriptor is None:
                son = t.right
                if son.left is None:
                    new_t = t.clone()
                    new_t.length_bp += son.length_bp
                    new_t.right = son.right
                    t = new_t

            t = t.update_sizes()
            
            assert (
                t.subtree_length_bp == old_subtree_length
            ), "Subtree length has changed after empty space optimization??"

            return t

        @staticmethod
        def expose(
            t: Optional['ScaffoldTree.Node'],
            from_bp: np.int64,
            to_bp: np.int64
        ) -> 'ScaffoldTree.ExposedSegment':
            total_length = t.subtree_length_bp if t is not None else np.int64(
                0)
            to_bp = min(to_bp, total_length)
            from_bp = max(np.int64(0), from_bp)
            le, gr = ScaffoldTree.Node.split_bp(
                t, to_bp, include_equal_to_the_left=True)
            le_size = (le.subtree_length_bp if le is not None else np.int64(0))
            # assert (
            #     le_size == to_bp
            # ), f"Less-or-equal part does not end where desired {le_size} != {to_bp}??"
            ls, sg = ScaffoldTree.Node.split_bp(
                le, from_bp, include_equal_to_the_left=True)
            less_size = (
                ls.subtree_length_bp if ls is not None else np.int64(0))
            segment_size = (
                sg.subtree_length_bp if sg is not None else np.int64(0))
            greater_size = (
                gr.subtree_length_bp if gr is not None else np.int64(0))

            # assert (
            #     less_size == from_bp
            # ), "Less size is not as requested??"

            # assert (
            #     (less_size + segment_size) == to_bp
            # ), "Less+Segment do not end as desired??"

            # assert (
            #     (less_size + segment_size + greater_size) == total_length
            # ), "Exposed segments do not sum up to the total length??"

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
    root_scaffold_id_counter: np.int64

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
        self.root_scaffold_id_counter = np.int64(0)

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
        if start_bp_incl > end_bp_excl:
            start_bp_incl, end_bp_excl = end_bp_excl, start_bp_incl

        with self.root_lock.gen_wlock():
            self.root_scaffold_id_counter += 1
            old_assembly_length_bp: np.int64 = self.root.subtree_length_bp
            es = ScaffoldTree.Node.expose(self.root, start_bp_incl, end_bp_excl)
            
            # le, gr = ScaffoldTree.Node.split_bp(self.root, end_bp_excl, include_equal_to_the_left=True)
            # ls, sg = ScaffoldTree.Node.split_bp(le, start_bp_incl, include_equal_to_the_left=False)            
            # es = ScaffoldTree.ExposedSegment(ls, sg, gr)
            
            assert (
                es.segment is not None
            ), "No segment corresponds to the requested borders"

            new_scaffold_node = ScaffoldTree.Node(
                length_bp=es.segment.subtree_length_bp,
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
            )

            self.commit_root(ScaffoldTree.ExposedSegment(
                less=es.less,
                segment=new_scaffold_node,
                greater=es.greater
            ))
            new_assembly_length_bp: np.int64 = self.root.subtree_length_bp
            assert (
                new_assembly_length_bp == old_assembly_length_bp
            ), "Assembly length changed after unscaffolding a region?"

    def traverse(self, fun: Callable[[Node], None]) -> None:
        with self.root_lock.gen_rlock():
            ScaffoldTree.Node.traverse(self.root, fun)

    def get_scaffold_at_bp(self, bp: np.int64) -> Optional[ScaffoldDescriptor]:
        with self.root_lock.gen_rlock():
            if bp >= self.root.subtree_length_bp or bp < 0:
                return None
            (l, r) = ScaffoldTree.Node.split_bp(
                self.root, bp, include_equal_to_the_left=False)
            assert (
                r is not None
            ), "Scaffold Tree root was None?"
            scaffold_node = ScaffoldTree.Node.leftmost(r)
            assert (
                scaffold_node is not None
            ), "Segment was not none but its leftmost is None"
            scaffold_node: ScaffoldTree.Node
            return scaffold_node.scaffold_descriptor

    def unscaffold(self, start_bp: np.int64, end_bp: np.int64) -> None:
        with self.root_lock.gen_wlock():
            old_assembly_length_bp: np.int64 = self.root.subtree_length_bp
            
            start_bp_extended, _, end_bp_extended, _ = self.extend_borders_to_scaffolds(
                start_bp, 
                end_bp
            )
            
            es = ScaffoldTree.Node.expose(self.root, start_bp_extended, end_bp_extended)
            empty_node = ScaffoldTree.Node(
                length_bp=es.segment.subtree_length_bp,
                scaffold_descriptor=None,
                y_priority=es.segment.y_priority,
                left=None,
                right=None,
                needs_changing_direction=False
            )
            self.commit_root(ScaffoldTree.ExposedSegment(
                less=es.less,
                segment=empty_node,
                greater=es.greater
            ))
            new_assembly_length_bp: np.int64 = self.root.subtree_length_bp
            assert (
                new_assembly_length_bp == old_assembly_length_bp
            ), "Assembly length changed after unscaffolding a region?"

    def rescaffold(self, start_bp: np.int64, end_bp: np.int64, spacer_length: Optional[int] = 1000) -> ScaffoldDescriptor:
        if start_bp > end_bp:
            start_bp, end_bp = end_bp, start_bp

        with self.root_lock.gen_wlock():
            self.root_scaffold_id_counter += 1
            old_assembly_length_bp: np.int64 = self.root.subtree_length_bp
            
            start_bp_extended, _, end_bp_extended, _ = self.extend_borders_to_scaffolds(
                start_bp, 
                end_bp
            )
            
            es = ScaffoldTree.Node.expose(self.root, start_bp_extended, end_bp_extended)
            assert (
                es.segment is not None
            ), f"No segment corresponds to the requested borders [{start_bp_extended}, {end_bp_extended}) with root of size {old_assembly_length_bp}"
            new_scaffold_descriptor = ScaffoldDescriptor.make_scaffold_descriptor(
                scaffold_id=self.root_scaffold_id_counter,
                scaffold_name=f"scaffold_auto_{self.root_scaffold_id_counter}_{datetime.datetime.now().strftime('%d-%M-%Y+%H:%M:%S')}",
                spacer_length=spacer_length
            )
            new_scaffold_node = ScaffoldTree.Node(
                length_bp=es.segment.subtree_length_bp,
                scaffold_descriptor=new_scaffold_descriptor,
                y_priority=es.segment.y_priority,
                left=None,
                right=None,
                needs_changing_direction=False
            )
            self.commit_root(ScaffoldTree.ExposedSegment(
                less=es.less,
                segment=new_scaffold_node,
                greater=es.greater
            ))
            new_assembly_length_bp: np.int64 = self.root.subtree_length_bp
            assert (
                new_assembly_length_bp == old_assembly_length_bp
            ), "Assembly length changed after unscaffolding a region?"
            return new_scaffold_descriptor

    def extend_borders_to_scaffolds(self, queried_start_bp: np.int64, queried_end_bp: np.int64) -> Tuple[np.int64, Optional[ScaffoldDescriptor], np.int64, Optional[ScaffoldDescriptor]]:
        with self.root_lock.gen_rlock():
            # Extend left border:
            opt_left_sd: Optional[ScaffoldDescriptor] = self.get_scaffold_at_bp(
                queried_start_bp)
            left_bp: np.int64 = queried_start_bp

            if opt_left_sd is not None:
                l, r = ScaffoldTree.Node.split_bp(
                    self.root, queried_start_bp, include_equal_to_the_left=False)
                left_bp = l.subtree_length_bp
                left_scaffold = ScaffoldTree.Node.leftmost(
                    r).scaffold_descriptor
                assert (
                    (left_scaffold == opt_left_sd) or (
                        (left_scaffold is None) != (opt_left_sd is None)
                    )
                ), "After extension of left selection border to the scaffold border, scaffold became different?"

            # Extend right border:
            opt_right_sd: Optional[ScaffoldDescriptor] = self.get_scaffold_at_bp(
                queried_end_bp)
            right_bp: np.int64 = queried_end_bp

            if opt_right_sd is not None:
                le, _ = ScaffoldTree.Node.split_bp(
                    self.root, queried_end_bp, include_equal_to_the_left=True)
                right_bp = le.subtree_length_bp
                right_scaffold = ScaffoldTree.Node.rightmost(
                    le).scaffold_descriptor
                assert (
                    (right_scaffold == opt_right_sd) or (
                        (right_scaffold is None) != (opt_right_sd is None)
                    )
                ), f"After extension of right selection border to the scaffold border, scaffold became different {right_scaffold} != {opt_right_sd}?"

            return left_bp, opt_left_sd, right_bp, opt_right_sd

    def move_selection_range(self, queried_start_bp: np.int64, queried_end_bp: np.int64, target_start_bp: np.int64) -> None:
        with self.root_lock.gen_wlock():
            left_bp, _, right_bp, _ = self.extend_borders_to_scaffolds(
                queried_start_bp, queried_end_bp)

            es = ScaffoldTree.Node.expose(
                self.root,
                from_bp=left_bp,
                to_bp=right_bp
            )

            tmp = ScaffoldTree.Node.merge(es.less, es.greater)
            nl, nr = ScaffoldTree.Node.split_bp(
                t=tmp,
                left_size=target_start_bp,
                include_equal_to_the_left=False,
            )

            self.commit_root(
                ScaffoldTree.ExposedSegment(
                    nl,
                    es.segment,
                    nr
                )
            )

    def get_scaffold_list(self) -> List[Tuple[ScaffoldDescriptor, ScaffoldBordersBP]]:
        descriptors: List[Tuple[ScaffoldDescriptor, ScaffoldBordersBP]] = []

        position: np.int64 = 0

        def traverse_fn(n: ScaffoldTree.Node) -> None:
            nonlocal position
            if n.scaffold_descriptor is not None:
                descriptors.append(
                    (n.scaffold_descriptor, ScaffoldBordersBP(position, position+n.length_bp)))
            position += n.length_bp

        self.traverse(traverse_fn)

        return descriptors
