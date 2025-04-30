
# pylint:disable=not-callable
import itertools
import math
import random
from collections.abc import Callable, Sequence
from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt  # For distinct colors
import numpy as np
import torch
from myai.transforms import tonumpy, totensor
from torch import nn

from ...benchmark import Benchmark
from .box_packing import CONTAINER1, uniform_container


def rotation_matrix(theta):
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t, cos_t], dim=-1)
    ], dim=-2)


def soft_point_in_polygon(points, vertices, smoothness=10.0):
    """
    Checks if points are inside a convex polygon defined by vertices.
    Assumes Y grows downwards (image coordinates) for grid and vertices.
    Args:
        points (torch.Tensor): Shape (H, W, 2) - Grid points coordinates (x, y).
        vertices (torch.Tensor): Shape (N_verts, 2) - Polygon vertices (x, y) in CCW order.
        smoothness (float): Controls the steepness of the sigmoid transition.
    Returns:
        torch.Tensor: Shape (H, W) - Soft indicator (0 to 1) for each point being inside.
    """
    vertices = vertices.float()
    points = points.float()

    edges = torch.roll(vertices, -1, dims=0) - vertices # N_verts, 2 - components dx, dy

    edge_lengths = torch.norm(edges, dim=1, keepdim=True)
    edge_lengths = torch.where(edge_lengths == 0, torch.ones_like(edge_lengths), edge_lengths)

    normals = torch.stack([edges[:, 1], -edges[:, 0]], dim=1) / edge_lengths # N_verts, 2

    points_exp = points.unsqueeze(2) # H, W, 1, 2
    vertices_exp = vertices.unsqueeze(0).unsqueeze(0) # 1, 1, N_verts, 2
    vec_to_point = points_exp - vertices_exp # H, W, N_verts, 2

    # signed distance - dot product of vector_to_point with the normal.
    signed_dist = torch.einsum('hwnc,nc->hwn', vec_to_point, normals) # H, W, N_verts

    inside_edge = torch.sigmoid(-smoothness * signed_dist) # H, W, N_verts

    inside_polygon, _ = torch.min(inside_edge, dim=-1) # H, W
    return inside_polygon

class SquishyBoxPacking(Benchmark):
    """
    box packing where it can apply affine to boxes but area stays the same.

    Args:
        boxes: sequence of (width, height) tuples for each box.
        container_size: tuple (W, H) of the container dimensions.
        render_resolution: tuple (res_H, res_W) for the internal rendering canvas.
        smoothness: box border sharpness.
        border_penalty_weight: weight for the loss term penalizing boxes outside container.
        overlap_penalty_weight: weight for the loss term penalizing overlapping boxes.
        init_strategy: how to initialize box positions ('center', 'random', 'random_inside').
    """
    def __init__(
        self,
        container_size=CONTAINER1[0],
        box_sizes=CONTAINER1[1],
        render_resolution: Sequence[int] = (128, 128),
        smoothness: float = 50.0,
        squishiness: float | None = 1,
        border_penalty_weight: float = 1.0,
        overlap_penalty_weight: float = 1.0,
        init_strategy: str = "random_inside",  # 'center', 'random', 'random_inside'
    ):
        super().__init__(log_projections=True)

        self.boxes = box_sizes
        self.num_boxes = len(box_sizes)
        self.container_W, self.container_H = container_size
        self.render_H, self.render_W = render_resolution
        self.smoothness = smoothness # Store the smoothness value
        self.border_penalty_weight = border_penalty_weight
        self.overlap_penalty_weight = overlap_penalty_weight
        self.squishiness = squishiness

        self.initial_areas = nn.Buffer(torch.tensor([w * h for w, h in box_sizes], dtype=torch.float32))
        self.initial_dims = nn.Buffer(torch.tensor(box_sizes, dtype=torch.float32))

        # init
        if init_strategy == 'center':
            init_translations = torch.tensor([[self.container_W / 2, self.container_H / 2]] * self.num_boxes)
        elif init_strategy == 'random':
            init_translations = torch.rand(self.num_boxes, 2) * torch.tensor([self.container_W, self.container_H])
        elif init_strategy == 'random_inside':
            buffer = 0.1
            rand_w = torch.rand(self.num_boxes, 1) * (self.container_W - buffer*2) + buffer
            rand_h = torch.rand(self.num_boxes, 1) * (self.container_H - buffer*2) + buffer
            init_translations = torch.cat([rand_w, rand_h], dim=1)
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        init_rotations = torch.zeros(self.num_boxes)
        init_squishes = torch.zeros(self.num_boxes)

        self.translations = nn.Parameter(init_translations)
        self.rotations = nn.Parameter(init_rotations)
        self.squishes = nn.Parameter(init_squishes) if squishiness is not None else nn.Buffer(init_squishes)

        # premake grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, self.container_H, self.render_H),
            torch.linspace(0, self.container_W, self.render_W),
            indexing='ij'
        )
        self.grid_points = nn.Buffer(torch.stack([grid_x, grid_y], dim=-1))

        cmap = plt.get_cmap('tab20')
        self.box_colors = [tuple(map(int, c)) for c in (np.array(cmap(np.linspace(0, 1, self.num_boxes)))[:, :3] * 255)]

        self.set_display_best('image')

    def _get_transformed_vertices(self):
        N = self.num_boxes
        half_dims = self.initial_dims / 2.0
        base_vertices = torch.stack([
            torch.stack([-half_dims[:, 0], -half_dims[:, 1]], dim=-1),
            torch.stack([ half_dims[:, 0], -half_dims[:, 1]], dim=-1),
            torch.stack([ half_dims[:, 0],  half_dims[:, 1]], dim=-1),
            torch.stack([-half_dims[:, 0],  half_dims[:, 1]], dim=-1)
        ], dim=1)

        sx = torch.exp(self.squishes).unsqueeze(-1).unsqueeze(-1)
        sy = torch.exp(-self.squishes).unsqueeze(-1).unsqueeze(-1)
        scaled_vertices = base_vertices * torch.cat([sx, sy], dim=-1)

        rot_matrices = rotation_matrix(self.rotations)
        rotated_vertices = torch.einsum('nij,nvj->nvi', rot_matrices, scaled_vertices)

        translations_exp = self.translations.unsqueeze(1)
        final_vertices = rotated_vertices + translations_exp

        return final_vertices

    def get_loss(self) -> torch.Tensor:
        """ Calculates the packing loss and generates a visualization frame. """
        transformed_vertices = self._get_transformed_vertices()

        canvas = torch.zeros((self.render_H, self.render_W), device=self.device, dtype=torch.float32)

        for i in range(self.num_boxes):
            box_canvas_i = soft_point_in_polygon(
                self.grid_points,
                transformed_vertices[i],
                self.smoothness
            )
            canvas += box_canvas_i

        # penalize overlap
        overlap_map = torch.relu(canvas - 1.0)
        overlap_loss = torch.mean(overlap_map) * self.overlap_penalty_weight

        # penalize border overlap
        verts = transformed_vertices
        verts_x, verts_y = verts[..., 0], verts[..., 1]
        loss_outside_x = torch.relu(verts_x - self.container_W) + torch.relu(-verts_x)
        loss_outside_y = torch.relu(verts_y - self.container_H) + torch.relu(-verts_y)
        border_loss = (torch.mean(loss_outside_x) + torch.mean(loss_outside_y)) * self.border_penalty_weight

        # penalize squishing
        squish_penalty = (self.squishes**2).mean() * (1/self.squishiness)

        total_loss = overlap_loss + border_loss + squish_penalty


        if self._make_images:
            frame = self._generate_frame(transformed_vertices)
            self.log('image', frame, log_test=False, to_uint8=False)

        return total_loss

    @torch.no_grad()
    def _generate_frame(self, vertices_tensor):
        vis_H, vis_W = 512, 512
        frame = np.zeros((vis_H, vis_W, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (vis_W - 1, vis_H - 1), (255, 255, 255), 1) # pylint:disable=no-member
        vertices_np = vertices_tensor.detach().cpu().numpy()

        for i in range(self.num_boxes):
            points = vertices_np[i].copy()
            points[:, 0] *= (vis_W / self.container_W)
            points[:, 1] *= (vis_H / self.container_H)
            points_int = points.astype(np.int32)

            color = self.box_colors[i % len(self.box_colors)]
            cv2.fillPoly(frame, [points_int], color) # pylint:disable=no-member

            outline_color = tuple(max(0, int(c * 0.6)) for c in color)
            cv2.polylines(frame, [points_int], isClosed=True, color=outline_color, thickness=1) # pylint:disable=no-member

        return frame

