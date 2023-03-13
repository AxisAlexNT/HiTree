from hict.core.common import ScaffoldDescriptor


class ScaffoldTree(object):
    class Node:
        length_bp: np.int64
        subtree_length_bp: np.int64
        # When scaffold_descriptor is None, node corresponds to the contigs not in scaffold
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
            if self.needs_changing_direction:
                (new_node.left, new_node.right) = (
                    new_node.right,
                    new_node.left
                )

        def update_sizes(self) -> 'ScaffoldTree.Node':
            new_node = self.clone()
            new_node.subtree_length_bp = length_bp
            if new_node.left is not None:
                new_node.subtree_length_bp += new_node.left.subtree_length_bp
            if new_node.right is not None:
                new_node.subtree_length_bp += new_node.right.subtree_length_bp
            return new_node

    root: ScaffoldTree.Node

    def __init__(
            self,
            assembly_length_bp: np.int64
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
            right=None
        )
