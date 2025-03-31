# Copyright (2025) critic-rl Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import string
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import pandas as pd


class NodeType(Enum):
    ROOT = "root"
    GENERATION = "generation"
    CRITIQUE_REVISION = "critique_revision"


@dataclass
class Node:
    task_id: int
    hash: str
    node_type: NodeType

    prev_hash: str = None
    solution: str = None
    critique: str = None
    success: str = None
    metadata: Dict = field(default_factory=lambda: {})

    def to_dict(self):
        return {
            k: v.value if isinstance(v, NodeType) else v
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Node":
        # Create a copy to avoid modifying the input
        node_data = data.copy()

        # Convert node_type string to NodeType enum if necessary
        if isinstance(node_data.get("node_type"), str):
            node_data["node_type"] = NodeType(node_data["node_type"])

        # Handle metadata field
        if "metadata" not in node_data:
            node_data["metadata"] = {}

        # Filter out any keys that aren't part of the Node class
        valid_fields = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in node_data.items() if k in valid_fields}

        return cls(**filtered_data)


class Tree:
    """Not exactly a tree, but a forest."""

    def __init__(self):
        self.nodes: List[List[Node]] = []

    @classmethod
    def generate_random_hash(cls) -> str:
        # Git uses 40-character SHA-1 hashes
        hash_length = 40
        # Valid hexadecimal characters
        hex_chars = string.hexdigits.lower()
        # Generate random hash
        return "".join(random.choice(hex_chars) for _ in range(hash_length))

    @classmethod
    def next_node_type(cls, node_type: NodeType) -> NodeType:
        return (
            NodeType.GENERATION
            if node_type == NodeType.ROOT
            else NodeType.CRITIQUE_REVISION
        )

    def add_nodes(self, nodes: List[Node], depth: int) -> None:
        assert depth >= 0 and depth <= self.max_depth + 1, "Invalid depth"
        if depth == self.max_depth + 1:
            self.nodes.append([])
        self.nodes[depth].extend(nodes)

    @property
    def num_roots(self) -> int:
        return len(self.nodes[0])

    @property
    def max_depth(self) -> int:
        return len(self.nodes) - 1

    @property
    def widths(self) -> List[int]:
        full_widths = [self.num_roots] + [len(nodes) for nodes in self.nodes]
        return [w // prev_w for w, prev_w in zip(full_widths[1:], full_widths)]

    def expand(self, new_widths: List[int]) -> Dict[int, List[Node]]:
        """Given a list of widths, return a dictionary of nodes to expand."""
        curr_widths = self.widths
        print(curr_widths, new_widths)

        # check if new_widths is valid
        if new_widths[0] != 1:
            raise ValueError("New widths must start with 1")
        if len(new_widths) < len(curr_widths):
            raise ValueError("New widths must be at least as long as current widths")
        if any(curr_width > new_widths[i] for i, curr_width in enumerate(curr_widths)):
            raise ValueError("New widths must be at least as large as current widths")

        # pad curr_widths with 0 (no nodes)
        while len(curr_widths) < len(new_widths):
            curr_widths.append(0)

        # expand new nodes
        # NOTE: we only return a layer of nodes that need to be expanded first
        # If no nodes to expand, return an empty dictionary
        new_nodes = {}
        for depth, (curr_width, new_width) in enumerate(zip(curr_widths, new_widths)):
            if new_width > curr_width:
                new_nodes[depth] = self.nodes[depth - 1] * (new_width - curr_width)
                break

        return new_nodes

    def expand_grouped(
        self, new_widths: List[int]
    ) -> Dict[int, Tuple[List[Node], int]]:
        """Given a list of widths, return a dictionary of nodes to expand.

        Note that instead of returning a dictionary of nodes to expand, this returns
        a dictionary of nodes to expand and the width for expansion.
        """
        curr_widths = self.widths
        print(curr_widths, new_widths)

        # check if new_widths is valid
        if new_widths[0] != 1:
            raise ValueError("New widths must start with 1")
        if len(new_widths) < len(curr_widths):
            raise ValueError("New widths must be at least as long as current widths")
        if any(curr_width > new_widths[i] for i, curr_width in enumerate(curr_widths)):
            raise ValueError("New widths must be at least as large as current widths")

        # pad curr_widths with 0 (no nodes)
        while len(curr_widths) < len(new_widths):
            curr_widths.append(0)

        # expand new nodes
        # NOTE: we only return a layer of nodes that need to be expanded first
        # If no nodes to expand, return an empty dictionary
        new_nodes = {}
        for depth, (curr_width, new_width) in enumerate(zip(curr_widths, new_widths)):
            if new_width > curr_width:
                new_nodes[depth] = self.nodes[depth - 1] * (new_width - curr_width)
                break

        return new_nodes

    def shuffle(self, seed: int = 42) -> None:
        random.seed(seed)
        for nodes in self.nodes:
            random.shuffle(nodes)

    def to_dataframe(self) -> pd.DataFrame:
        all_nodes = sum(self.nodes, [])
        return pd.DataFrame([n.to_dict() for n in all_nodes])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Tree":
        records = df.to_dict(orient="records")
        nodes_flatten = [Node.from_dict(record) for record in records]
        hash_to_depth = {}
        depth_to_nodes = defaultdict(list)

        for node in nodes_flatten:
            hash = node.hash
            prev_hash = node.prev_hash

            if prev_hash is None:  # root
                depth_to_nodes[0].append(node)
                hash_to_depth[hash] = 0
            else:
                assert prev_hash in hash_to_depth, "Nodes not sorted by depths"
                prev_depth = hash_to_depth[prev_hash]
                depth_to_nodes[prev_depth + 1].append(node)
                hash_to_depth[hash] = prev_depth + 1

        tree = cls()
        for depth in sorted(depth_to_nodes.keys()):
            tree.add_nodes(depth_to_nodes[depth], depth)
        return tree
