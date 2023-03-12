from hict.core.common import ScaffoldDescriptor


class ScaffoldTree(object):
    class Node:
        length_bp: np.int64
        scaffold_descriptor: Optional[ScaffoldDescriptor]
        y_priority: np.int64
        left: Optional['ScaffoldTree.Node']
        right: Optional['ScaffoldTree.Node']

        def __init__(
            self,
            length_bp: np.int64,
            scaffold_descriptor: Optional[ScaffoldDescriptor],
            y_priority: np.int64,
            left: Optional['ScaffoldTree.Node'],
            right: Optional['ScaffoldTree.Node'],
        ):
            self.length_bp = length_bp
            self.scaffold_descriptor = scaffold_descriptor
            self.y_priority = y_priority
            self.left = left
            self.right = right

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
