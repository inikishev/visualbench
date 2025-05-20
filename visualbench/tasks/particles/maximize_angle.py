# pylint:disable=no-member
import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from ...benchmark import Benchmark



class MaximizeSmallestAngle(Benchmark):
    def __init__(self, n, resolution = (512, 512), penalty=10.0, seed=0):
        super().__init__(seed=seed)
        if n < 3: raise ValueError("Need at least 3 points to form a triangle.")
        self.n = n
        self.image_width, self.image_height = resolution
        self.penalty = penalty
        self.points = nn.Parameter(torch.rand(n, 2, generator=self.rng.torch()) / 2 + 0.25)

    def _calculate_angles_for_triangle(self, p0, p1, p2):
        v01 = p1 - p0
        v02 = p2 - p0
        v10 = p0 - p1
        v12 = p2 - p1
        v20 = p0 - p2
        v21 = p1 - p2

        norm_v01 = torch.norm(v01, p=2, dim=-1) + 1e-9
        norm_v02 = torch.norm(v02, p=2, dim=-1) + 1e-9
        norm_v12 = torch.norm(v12, p=2, dim=-1) + 1e-9

        norm_v10 = norm_v01
        norm_v20 = norm_v02
        norm_v21 = norm_v12

        dot_p0 = torch.sum(v01 * v02, dim=-1)
        dot_p1 = torch.sum(v10 * v12, dim=-1)
        dot_p2 = torch.sum(v20 * v21, dim=-1)

        cos_angle0 = torch.clamp(dot_p0 / (norm_v01 * norm_v02), -1.0 + 1e-7, 1.0 - 1e-7)
        cos_angle1 = torch.clamp(dot_p1 / (norm_v10 * norm_v12), -1.0 + 1e-7, 1.0 - 1e-7)
        cos_angle2 = torch.clamp(dot_p2 / (norm_v20 * norm_v21), -1.0 + 1e-7, 1.0 - 1e-7)

        angle0 = torch.acos(cos_angle0)
        angle1 = torch.acos(cos_angle1)
        angle2 = torch.acos(cos_angle2)

        return torch.stack([angle0, angle1, angle2], dim=-1) # (N, 3) or (3,)

    def get_loss(self):
        # 1. Form all possible triangles
        indices_combiner = torch.combinations(torch.arange(self.n, device=self.device), 3)

        p0 = self.points[indices_combiner[:, 0]]
        p1 = self.points[indices_combiner[:, 1]]
        p2 = self.points[indices_combiner[:, 2]]

        # 2. Calculate all angles for all triangles
        all_angles = self._calculate_angles_for_triangle(p0, p1, p2)

        # 3. Find the smallest angle among all angles
        min_angle, idx = torch.min(all_angles.view(-1), dim=0)
        critical_triangle_idx = idx // 3

        # 4. Objective
        loss = -min_angle

        # 5. Penalty for going out of bounds [0, 1]
        penalty = 0
        # Penalty for x < 0 or x > 1
        penalty += torch.sum(torch.relu(-self.points[:, 0])**2) # pylint:disable = invalid-unary-operand-type
        penalty += torch.sum(torch.relu(self.points[:, 0] - 1.0)**2) # pylint:disable = invalid-unary-operand-type
        # Penalty for y < 0 or y > 1
        penalty += torch.sum(torch.relu(-self.points[:, 1])**2) # pylint:disable = invalid-unary-operand-type
        penalty += torch.sum(torch.relu(self.points[:, 1] - 1.0)**2)

        loss = loss + self.penalty * penalty

        # 6. Generate image
        if self._make_images:
            frame = self._generate_frame(min_angle.item(), (idx % 3).item(), indices_combiner[critical_triangle_idx].tolist())
            self.log_image('points', frame, to_uint8=False, show_best=True)

        return loss

    @torch.no_grad
    def _generate_frame(self, min_angle, vertex_idx, triangle_idx):
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        # Scale normalized points to image dimensions for visualization
        points = self.points.detach() # pylint:disable=not-callable
        points_scaled = torch.zeros_like(points)
        points_scaled[:, 0] = points[:, 0] * self.image_width
        points_scaled[:, 1] = points[:, 1] * self.image_height
        points_scaled = points_scaled.detach().cpu().numpy().astype(int)


        # Draw all points
        for i in range(self.n):
            pt = (points_scaled[i, 0], points_scaled[i, 1])
             # Clip points to be within image boundaries for drawing, just in case penalty isn't enough
            pt = (np.clip(pt[0], 0, self.image_width -1), np.clip(pt[1], 0, self.image_height -1))
            cv2.circle(img, pt, 3, (255, 100, 100), -1) # Light blue points
            cv2.putText(img, str(i), (pt[0]+5, pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        # Draw all triangles faintly
        if self.n >= 3:
            indices_combiner = torch.combinations(torch.arange(self.n), 3) # on CPU
            for combo_idx in range(indices_combiner.shape[0]):
                idx0, idx1, idx2 = indices_combiner[combo_idx]
                pt0 = tuple(np.clip(points_scaled[idx0], [0,0], [self.image_width-1, self.image_height-1]))
                pt1 = tuple(np.clip(points_scaled[idx1], [0,0], [self.image_width-1, self.image_height-1]))
                pt2 = tuple(np.clip(points_scaled[idx2], [0,0], [self.image_width-1, self.image_height-1]))

                cv2.line(img, pt0, pt1, (100, 100, 100), 1)
                cv2.line(img, pt1, pt2, (100, 100, 100), 1)
                cv2.line(img, pt2, pt0, (100, 100, 100), 1)

        # Highlight the critical triangle and angle
        if (min_angle is not None and
            triangle_idx is not None and
            vertex_idx is not None and
            len(triangle_idx) == 3): # Ensure valid cache

            p_indices = triangle_idx # [idx_A, idx_B, idx_C]

            # Points of the critical triangle
            pt_coords_draw = [tuple(np.clip(points_scaled[i], [0,0], [self.image_width-1, self.image_height-1])) for i in p_indices]


            cv2.line(img, pt_coords_draw[0], pt_coords_draw[1], (0, 255, 255), 2) # Yellow
            cv2.line(img, pt_coords_draw[1], pt_coords_draw[2], (0, 255, 255), 2)
            cv2.line(img, pt_coords_draw[2], pt_coords_draw[0], (0, 255, 255), 2)

            vertex_orig_idx = p_indices[vertex_idx]
            other_indices_in_triangle = [i for i in range(3) if i != vertex_idx]

            p_vertex_draw = np.clip(points_scaled[vertex_orig_idx], [0,0], [self.image_width-1, self.image_height-1])
            p_side1_end_draw = np.clip(points_scaled[p_indices[other_indices_in_triangle[0]]], [0,0], [self.image_width-1, self.image_height-1])
            p_side2_end_draw = np.clip(points_scaled[p_indices[other_indices_in_triangle[1]]], [0,0], [self.image_width-1, self.image_height-1])

            cv2.line(img, tuple(p_vertex_draw), tuple(p_side1_end_draw), (0, 0, 255), 2) # Red
            cv2.line(img, tuple(p_vertex_draw), tuple(p_side2_end_draw), (0, 0, 255), 2) # Red
            cv2.circle(img, tuple(p_vertex_draw), 4, (0,0,255), -1) # Red filled circle

            angle_deg = math.degrees(min_angle)
            cv2.putText(img, f"Min Angle: {angle_deg:.2f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(img, "Optimizing...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return img[:,:,::-1]
