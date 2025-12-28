import random
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from torch import nn

from ..benchmark import Benchmark


def _complete_graph(n: int = 20) -> list[list[int]]:
    """Generates a complete graph K_n."""
    if n <= 0: return []
    adj = [[] for _ in range(n)]
    if n == 1: return adj
    for i in range(n):
        for j in range(i + 1, n):
            adj[i].append(j)
            adj[j].append(i)
    return adj

def _grid_graph(rows: int = 8, cols: int = 8) -> list[list[int]]:
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

def _barbell_graph(clique_size: int = 10) -> list[list[int]]:
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


def _watts_strogatz_graph(n: int = 30, k: int = 4, p: float = 0.5) -> list[list[int]]:
    """Generates a Watts-Strogatz small-world graph."""
    generator = random.Random(0)

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
            if generator.random() < p:
                original_neighbor = neighbor
                # choose a new node w != i and w not already connected to i
                possible_new_neighbors = [w for w in nodes if w != i and w not in adj[i]]

                if possible_new_neighbors: # check if there's anyone left to rewire to
                    new_neighbor = generator.choice(possible_new_neighbors)

                    # rewire: remove old edge, add new edge
                    adj[i].remove(original_neighbor)
                    adj[original_neighbor].remove(i)
                    adj[i].add(new_neighbor)
                    adj[new_neighbor].add(i)
                # else: cannot rewire this edge as i is connected to everyone else

    # convert sets back to lists
    adj_list = [sorted(list(neighbors)) for neighbors in adj]
    return adj_list



class GraphLayout(Benchmark):
    """Optimize graph layout by edge attraction and node repulsion.

    Renders:
        current graph layout.

    Args:
        adj (Sequence[Sequence[int]]): Adjacency list representation of the graph.
                            adj[i] contains the list of neighbors for node i.
        k_attraction (float): Strength of the attraction force between connected nodes.
        k_repulsion (float): Strength of the repulsion force between all nodes.
        epsilon (float): Small value added to distances to prevent division by zero.
        init_pos (Optional[np.ndarray]): Optional initial positions for nodes (shape: [num_nodes, 2]).
                                        If None, random positions are used.
        node_radius (int): Radius of nodes in visualization.
        line_thickness (int): Thickness of edges in visualization.
        resolution (int): The width and height of the visualization canvas.
        camera_smoothing_factor (float): Smoothing factor for camera movement (0 < alpha <= 1).
                                        Smaller values result in smoother, slower camera motion.
    """
    COMPLETE = staticmethod(_complete_graph)
    GRID = staticmethod(_grid_graph)
    BARBELL = staticmethod(_barbell_graph)
    WATTS_STROGATZ = staticmethod(_watts_strogatz_graph)
    def __init__(
        self,
        adj: Sequence[Sequence[int]],
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
        resolution: int = 400,
        camera_smoothing_factor: float = 0.2,
    ):
        super().__init__()

        num_nodes = len(adj)
        if not all(isinstance(neighbors, Sequence) for neighbors in adj):
            raise ValueError("Elements of adj must be lists or tuples")

        self.num_nodes = num_nodes
        self.adj = adj
        self.canvas_size = resolution
        self.k_attraction = k_attraction
        self.k_repulsion = k_repulsion
        self.epsilon = epsilon
        self._make_images = make_images
        self.camera_smoothing_factor = camera_smoothing_factor

        self.node_radius = node_radius
        self.line_thickness = line_thickness
        self.node_color = tuple(reversed(node_color)) # because its BGR
        self.edge_color = tuple(reversed(edge_color))
        self.bg_color = tuple(reversed(bg_color))

        # node positions
        if init_pos is None:
            positions = torch.rand(num_nodes, 2, dtype=torch.float32, generator=self.rng.torch()) * resolution
        else:
            positions = torch.as_tensor(init_pos, dtype=torch.float32)

        self.node_positions = nn.Parameter(positions)

        # camera state initialization
        with torch.no_grad():
            self.camera_center = nn.Buffer(torch.mean(self.node_positions, dim=0) if num_nodes > 0 else torch.zeros(2))
            if num_nodes > 1:
                spread = torch.max(self.node_positions, dim=0).values - torch.min(self.node_positions, dim=0).values
                max_spread = torch.max(spread)
                # set initial scale to fit all nodes with a 10% margin
                initial_scale = (self.canvas_size * 0.9) / (max_spread + self.epsilon)
            else:
                initial_scale = torch.tensor(1.0)
            self.camera_scale = nn.Buffer(initial_scale)

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

        self.set_multiobjective_func(torch.sum)
        self._show_titles_on_video = False

    def get_loss(self) -> torch.Tensor:
        pos = self.node_positions

        # attraction loss
        attraction = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
        if self.edges:
            edge_nodes_u = pos[[u for u, v in self.edges]] # num_edges, 2
            edge_nodes_v = pos[[v for u, v in self.edges]] # num_edges, 2
            diff = edge_nodes_u - edge_nodes_v
            attraction = torch.mean(diff * diff) # num_edges
        attraction = self.k_attraction * attraction

        # repulsion loss
        repulsion = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
        if self.num_nodes > 1:
            # distances between all pairs (i, j)
            dist_sq = torch.mean((pos.unsqueeze(1) - pos.unsqueeze(0)) ** 2, dim=-1)
            inv_dist_sq = 1.0 / (dist_sq + self.epsilon)
            # no self repulsion
            inv_dist_sq.fill_diagonal_(0)
            repulsion = self.k_repulsion * torch.mean(inv_dist_sq) / 2.0

        # visualize
        if self._make_images:
            # update camera state smoothly before rendering
            self._update_camera()
            frame = self._make_frame(self.node_positions.detach().cpu().numpy()) # pylint:disable=not-callable
            self.log_image('graph', frame, to_uint8=False, show_best=True)

        return torch.stack([attraction, repulsion])

    @torch.no_grad
    def _update_camera(self):
        """Updates camera center and scale using an exponential moving average for smoothness."""
        if self.num_nodes == 0:
            return

        pos = self.node_positions
        alpha = self.camera_smoothing_factor

        # 1. Smoothly update camera center to the mean of node positions
        target_center = torch.mean(pos, dim=0).nan_to_num_(0, self.canvas_size, -self.canvas_size)
        self.camera_center.lerp_(target_center, 1-alpha).nan_to_num_(0, self.canvas_size, -self.canvas_size)


        # 2. Calculate target scale to fit all nodes based on the new camera center
        if self.num_nodes > 1:
            # find the furthest point from the new (smoothed) camera center
            relative_pos = pos - self.camera_center
            max_dist_from_center = torch.max(torch.sqrt(torch.sum(relative_pos**2, dim=1)))
            # target scale should fit this point with a margin (e.g., 90% of half the canvas)
            margin_factor = 0.9
            target_scale = (self.canvas_size / 2 * margin_factor) / (max_dist_from_center + self.epsilon)
        else:
            # for a single node, a default scale is fine
            target_scale = torch.tensor(1.0, device=self.camera_scale.device, dtype=self.camera_scale.dtype)

        # 3. Smoothly update camera scale
        self.camera_scale = alpha * target_scale + (1 - alpha) * self.camera_scale

    @torch.no_grad
    def _make_frame(self, pos: np.ndarray) -> np.ndarray:
        """Renders the graph from the camera's perspective."""
        canvas = np.full((self.canvas_size, self.canvas_size, 3), self.bg_color, dtype=np.uint8)
        if self.num_nodes == 0:
            return canvas

        # get current camera state
        center_np = self.camera_center.cpu().numpy()
        scale_np = self.camera_scale.cpu().numpy()
        canvas_center = np.array([self.canvas_size / 2, self.canvas_size / 2])

        # transform node positions from world space to screen space
        # 1. Translate to origin (relative to camera center)
        # 2. Scale (zoom)
        # 3. Translate to canvas center
        screen_pos = (pos - center_np) * scale_np + canvas_center
        screen_pos = np.nan_to_num(screen_pos, nan=0, posinf=0, neginf=0).clip(-1e5,1e5).astype(int)

        # draw edges
        if self.edges:
            for u, v in self.edges:
                pt1 = tuple(screen_pos[u])
                pt2 = tuple(screen_pos[v])

                cv2.line(canvas, pt1, pt2, self.edge_color, self.line_thickness, lineType=cv2.LINE_AA) # pylint:disable=no-member

        # draw nodes
        for i in range(self.num_nodes):
            center = tuple(screen_pos[i])
            cv2.circle(canvas, center, self.node_radius, self.node_color, -1, lineType=cv2.LINE_AA) # pylint:disable=no-member

        return canvas