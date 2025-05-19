import random
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from torch import nn

from ..benchmark import Benchmark


class GraphLayout(Benchmark):
    """
    Optimize graph layout by edge attraction and node repulsion.

    Args:
        adj (List[List[int]]): Adjacency list representation of the graph.
                            adj[i] contains the list of neighbors for node i.
        canvas_size (int): The width and height of the visualization canvas.
        k_attraction (float): Strength of the attraction force between connected nodes.
        k_repulsion (float): Strength of the repulsion force between all nodes.
        epsilon (float): Small value added to distances to prevent division by zero.
        init_pos (Optional[np.ndarray]): Optional initial positions for nodes (shape: [num_nodes, 2]).
                                        If None, random positions are used.
        node_radius (int): Radius of nodes in visualization.
        line_thickness (int): Thickness of edges in visualization.
    """

    def __init__(
        self,
        adj: Sequence[Sequence[int]],
        canvas_size: int = 400,
        k_attraction: float = 1.0,
        k_repulsion: float = 1e7,
        epsilon: float = 1e-4,
        init_pos: np.ndarray | None = None,
        make_images: bool = True,
        node_radius: int = 5,
        line_thickness: int = 1,
        node_color: tuple[int, int, int] = (255, 0, 0),
        edge_color: tuple[int, int, int] = (0, 255, 0),
        bg_color: tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__()

        num_nodes = len(adj)
        if not all(isinstance(neighbors, Sequence) for neighbors in adj):
            raise ValueError("Elements of adj must be lists or tuples")

        self.num_nodes = num_nodes
        self.adj = adj
        self.canvas_size = canvas_size
        self.k_attraction = k_attraction
        self.k_repulsion = k_repulsion
        self.epsilon = epsilon
        self._make_images = make_images

        self.node_radius = node_radius
        self.line_thickness = line_thickness
        self.node_color = tuple(reversed(node_color)) # because its BGR
        self.edge_color = tuple(reversed(edge_color))
        self.bg_color = tuple(reversed(bg_color))

        # node positions
        if init_pos is None:
            positions = torch.rand(num_nodes, 2, dtype=torch.float32) * canvas_size
        else:
            positions = torch.as_tensor(init_pos, dtype=torch.float32)

        self.node_positions = nn.Parameter(positions)

        self.frames: list[np.ndarray] = []

        # precompute and store edges as (u, v) pairs where u < v to avoid duplicates
        self.edges: list[tuple[int, int]] = []
        nodes_with_edges = set()
        for u, neighbors in enumerate(self.adj):
            for v in neighbors:
                if not 0 <= v < self.num_nodes:
                    raise ValueError(f"Node index {v} in adjacency list for node {u} is out of bounds [0, {num_nodes-1}]")
                if u < v:
                    self.edges.append((u, v))
            if neighbors:
                nodes_with_edges.add(u)
                nodes_with_edges.update(neighbors)

        # checks
        if not self.edges and num_nodes > 1:
            print("Warning: No edges found in the adjacency list. Attraction loss will be zero.")
        elif len(nodes_with_edges) < num_nodes and num_nodes > 0 :
            print(f"Warning: {num_nodes - len(nodes_with_edges)} nodes appear to have no edges. Attraction loss will not affect them.")

    def get_loss(self) -> torch.Tensor:
        pos = self.node_positions
        pos_clamped = torch.clip(pos, 0, self.canvas_size - 1)
        penalty = torch.mean((pos - pos_clamped) ** 2)

        # attraction loss
        attraction = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
        if self.edges:

            edge_nodes_u = pos[ [u for u, v in self.edges] ] # num_edges, 2
            edge_nodes_v = pos[ [v for u, v in self.edges] ] # num_edges, 2

            diff = edge_nodes_u - edge_nodes_v
            attraction = torch.mean(diff * diff) # num_edges

        attraction = self.k_attraction * attraction

        # repulsion loss
        repulsion = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
        if self.num_nodes > 1:
            # istances between all pairs (i, j)
            dist_sq = torch.mean((pos.unsqueeze(1) - pos.unsqueeze(0)) ** 2, dim=-1)
            inv_dist_sq = 1.0 / (dist_sq + self.epsilon)

            # no self repulsion
            inv_dist_sq = inv_dist_sq.fill_diagonal_(0)

            repulsion = self.k_repulsion * torch.mean(inv_dist_sq) / 2.0

        # loss
        loss = attraction + repulsion + penalty

        # visualize
        if self._make_images:
            frame = self._make_frame(self.node_positions.detach().cpu().numpy()) # pylint:disable=not-callable ???
            self.log_image('graph', frame, to_uint8=False, show_best=True)

        return loss

    @torch.no_grad
    def _make_frame(self, pos: np.ndarray) -> np.ndarray:
        canvas = np.full((self.canvas_size, self.canvas_size, 3), self.bg_color, dtype=np.uint8)

        pos_np_clamped = np.clip(pos, 0, self.canvas_size - 1).astype(int)

        if self.edges:
            for u, v in self.edges:
                pt1 = tuple(pos_np_clamped[u])
                pt2 = tuple(pos_np_clamped[v])
                cv2.line(canvas, pt1, pt2, self.edge_color, self.line_thickness, lineType=cv2.LINE_AA) # pylint:disable=no-member

        for i in range(self.num_nodes):
            center = tuple(pos_np_clamped[i])
            cv2.circle(canvas, center, self.node_radius, self.node_color, -1, lineType=cv2.LINE_AA) # pylint:disable=no-member

        return canvas


def complete_graph(n: int = 20) -> list[list[int]]:
    """Generates a complete graph K_n."""
    if n <= 0: return []
    adj = [[] for _ in range(n)]
    if n == 1: return adj
    for i in range(n):
        for j in range(i + 1, n):
            adj[i].append(j)
            adj[j].append(i)
    return adj

def grid_graph(rows: int = 8, cols: int = 8) -> list[list[int]]:
    """Generates an m x n grid graph."""
    if rows <= 0 or cols <= 0: return []
    n = rows * cols
    adj = [[] for _ in range(n)]
    for r in range(rows):
        for c in range(cols):
            index = r * cols + c
            # connect to right neighbor
            if c + 1 < cols:
                right_index = index + 1
                adj[index].append(right_index)
                adj[right_index].append(index)
            # connect to bottom neighbor
            if r + 1 < rows:
                bottom_index = index + cols
                adj[index].append(bottom_index)
                adj[bottom_index].append(index)
    return adj

def barbell_graph(clique_size: int = 10) -> list[list[int]]:
    """Generates a barbell graph: two K_m cliques connected by a single edge."""
    if clique_size <= 0: return []
    if clique_size == 1:
        return [[1], [0]]

    n = 2 * clique_size
    adj = [[] for _ in range(n)]

    # first clique (nodes 0 to clique_size - 1)
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            adj[i].append(j)
            adj[j].append(i)

    # second clique (nodes clique_size to 2*clique_size - 1)
    for i in range(clique_size, n):
        for j in range(i + 1, n):
            adj[i].append(j)
            adj[j].append(i)

    # connecting edge (connect last node of first clique to first node of second clique)
    node1 = clique_size - 1
    node2 = clique_size
    adj[node1].append(node2)
    adj[node2].append(node1)

    return adj


def watts_strogatz_graph(n: int = 30, k: int = 4, p: float = 0.5) -> list[list[int]]:
    """Generates a Watts-Strogatz small-world graph."""
    if k % 2 != 0 or k >= n:
        raise ValueError("k must be an even integer less than n")
    if not 0 <= p <= 1:
        raise ValueError("p (rewiring probability) must be between 0 and 1")
    if n <= 0: return []

    adj = [set() for _ in range(n)]

    # 1. create ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n
            adj[i].add(neighbor)
            adj[neighbor].add(i)

    # 2. rewire edges
    nodes = list(range(n))
    for i in range(n):
        # only rewire edges to the k/2 clockwise neighbors
        neighbors_to_consider = [(i + j) % n for j in range(1, k // 2 + 1)]

        for neighbor in neighbors_to_consider:
            if random.random() < p:
                original_neighbor = neighbor
                # choose a new node w != i and w not already connected to i
                possible_new_neighbors = [w for w in nodes if w != i and w not in adj[i]]

                if possible_new_neighbors: # check if there's anyone left to rewire to
                    new_neighbor = random.choice(possible_new_neighbors)

                    # rewire: remove old edge, add new edge
                    adj[i].remove(original_neighbor)
                    adj[original_neighbor].remove(i)
                    adj[i].add(new_neighbor)
                    adj[new_neighbor].add(i)
                # else: cannot rewire this edge as i is connected to everyone else

    # convert sets back to lists
    adj_list = [sorted(list(neighbors)) for neighbors in adj]
    return adj_list
