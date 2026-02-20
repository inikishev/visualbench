# pylint:disable=no-member
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import totensor

# Circular obstacles: (y, x, radius)
DEFAULT_OBSTACLES = (
    (5.0, 8.0, 1.5),
    (9.0, 12.0, 1.5),
    (4.0, 15.0, 1.0),
    (10.0, 5.0, 1.0),
)

DEFAULT_INIT = (7.0, 2.0)
DEFAULT_FOOD = (7.0, 12.0)


class Snake(Benchmark):
    """
    Snake game benchmark. The goal is to navigate a snake to collect food
    while avoiding circular obstacles and the snake's own body.

    The snake moves in a continuous space towards food while avoiding
    circular obstacles that can be navigated around smoothly.

    Args:
        obstacles: List of (y, x, radius) tuples for circular obstacles
        init: Initial snake position (y, x)
        food: Initial food position (y, x)
        cell_size: Size of each cell in pixels for rendering
        snake_length: Target length of the snake (number of food to collect)
        arena_size: (height, width) of the arena.
        body_history_threshold: Minimum distance to add new body segment. Defaults to 0.3.
        food_collection_distance: Distance to collect food. Defaults to 0.6.
        direction_step_size: Step size towards food. Defaults to 0.5.
        self_collision_threshold: Distance for self-collision penalty. Defaults to 0.6.
        obstacle_penalty_multiplier: Obstacle collision penalty weight. Defaults to 50.
        boundary_penalty_multiplier: Boundary violation penalty weight. Defaults to 10.
        obstacle_margin: Snake radius for obstacle collision. Defaults to 0.5.
        food_spawn_obstacle_margin: Minimum distance from obstacles when spawning food. Defaults to 1.0.
        food_spawn_body_offset: Extra body segments to keep in history. Defaults to 5.
        food_spawn_angle_increment: Angle increment per food collected. Defaults to 0.8.
        food_spawn_angle_attempt_increment: Angle increment per spawn attempt. Defaults to 0.1.
        food_spawn_base_radius: Base radius for food spawn spiral. Defaults to 3.
        food_spawn_radius_increment: Radius increment per food collected. Defaults to 0.5.
        food_spawn_max_attempts: Maximum attempts to find valid food spawn position. Defaults to 10.
        boundary_margin_inner: Inner boundary margin for food spawning. Defaults to 1.
        boundary_margin_outer: Outer boundary margin for boundary loss. Defaults to 1.5.
        boundary_loss_offset: Offset for boundary loss calculation. Defaults to 0.5.
        epsilon: Small value for numerical stability. Defaults to 1e-8.

    """

    def __init__(
        self,
        obstacles=DEFAULT_OBSTACLES,
        init=DEFAULT_INIT,
        food=DEFAULT_FOOD,
        cell_size=24,
        snake_length=10,
        arena_size=(15, 20),

        # Hyperparameters
        body_history_threshold = 0.3,
        food_collection_distance = 0.6,
        direction_step_size = 0.5,
        self_collision_threshold = 0.6,
        obstacle_penalty_multiplier = 50.0,
        boundary_penalty_multiplier = 10.0,
        obstacle_margin = 0.5,
        food_spawn_obstacle_margin = 1.0,
        food_spawn_body_offset = 5,
        food_spawn_angle_increment = 0.8,
        food_spawn_angle_attempt_increment = 0.1,
        food_spawn_base_radius = 3,
        food_spawn_radius_increment = 0.5,
        food_spawn_max_attempts = 10,
        boundary_margin_inner = 1,
        boundary_margin_outer = 1.5,
        boundary_loss_offset = 0.5,
        epsilon = 1e-8,
    ):

        super().__init__()
        self.h, self.w = arena_size

        self.body_history_threshold = body_history_threshold
        self.food_collection_distance = food_collection_distance
        self.direction_step_size = direction_step_size
        self.self_collision_threshold = self_collision_threshold
        self.obstacle_penalty_multiplier = obstacle_penalty_multiplier
        self.boundary_penalty_multiplier = boundary_penalty_multiplier
        self.obstacle_margin = obstacle_margin
        self.food_spawn_obstacle_margin = food_spawn_obstacle_margin
        self.food_spawn_body_offset = food_spawn_body_offset
        self.food_spawn_angle_increment = food_spawn_angle_increment
        self.food_spawn_angle_attempt_increment = food_spawn_angle_attempt_increment
        self.food_spawn_base_radius = food_spawn_base_radius
        self.food_spawn_radius_increment = food_spawn_radius_increment
        self.food_spawn_max_attempts = food_spawn_max_attempts
        self.boundary_margin_inner = boundary_margin_inner
        self.boundary_margin_outer = boundary_margin_outer
        self.boundary_loss_offset = boundary_loss_offset
        self.epsilon = epsilon

        # Store obstacles as registered buffer
        self.obstacles = nn.Buffer(totensor(obstacles, dtype=torch.float32))

        self.cell_size = cell_size
        self.target_length = snake_length

        # Snake position (continuous) - optimized directly
        self.position = nn.Parameter(totensor(init, dtype=torch.float32))

        # Food position (registered buffer that can be updated)
        self.food = nn.Buffer(totensor(food, dtype=torch.float32))

        # Track collected food (registered buffer)
        self.collected_food = nn.Buffer(torch.tensor(0, dtype=torch.int32))

        # Track snake body history
        self.body_history = []
        self.food_history = []

        # Pre-render background
        self.bg_image = self._render_background()

        # Buffers for pre_step -> get_loss communication
        self.target_ = None
        self.obstacle_loss_ = None
        self.self_collision_loss_ = None
        self.dist_to_food_ = None

    def reset(self):
        self.body_history = []
        self.food_history = []
        self.target_ = None
        self.obstacle_loss_ = None
        self.self_collision_loss_ = None
        self.dist_to_food_ = None
        return super().reset()

    def pre_step(self):
        # Store previous position for body tracking (only if moved significantly)
        with torch.no_grad():
            if len(self.body_history) == 0:
                self.body_history.append(self.position.detach().cpu().numpy().copy()) # pylint:disable=not-callable
            else:
                last_pos = torch.tensor(self.body_history[-1], dtype=torch.float32, device=self.position.device)
                if torch.norm(self.position - last_pos) > self.body_history_threshold:
                    self.body_history.append(self.position.detach().cpu().numpy().copy()) # pylint:disable=not-callable

            # Keep only last N positions based on snake length
            max_len = int(self.collected_food.item()) + self.food_spawn_body_offset
            self.body_history = self.body_history[-max_len:]

            # Check if food is collected
            new_x = new_y = None
            dist_to_food = torch.norm(self.position - self.food)
            if dist_to_food < self.food_collection_distance:
                self.collected_food.add_(1)
                # Move food to new position (spiral pattern from center)
                # Make sure it doesn't spawn inside obstacles
                for attempt in range(self.food_spawn_max_attempts):
                    angle = self.collected_food.item() * self.food_spawn_angle_increment + attempt * self.food_spawn_angle_attempt_increment
                    radius = self.food_spawn_base_radius + self.collected_food.item() * self.food_spawn_radius_increment
                    new_y = self.h / 2 + torch.cos(torch.tensor(angle, dtype=torch.float32)).item() * radius
                    new_x = self.w / 2 + torch.sin(torch.tensor(angle, dtype=torch.float32)).item() * radius

                    # Check distance to all obstacles
                    valid = True
                    for obs in self.obstacles:
                        obs_y, obs_x, obs_r = obs
                        dist_to_obs = torch.norm(torch.tensor([new_y, new_x], dtype=torch.float32) - torch.tensor([obs_y.item(), obs_x.item()], dtype=torch.float32))
                        if dist_to_obs < obs_r.item() + self.food_spawn_obstacle_margin:
                            valid = False
                            break

                    if valid:
                        break

                # Clamp to arena bounds (stay inside boundaries)
                assert new_x is not None and new_y is not None
                new_y = float(max(self.boundary_margin_inner, min(self.h - self.boundary_margin_outer - 0.5, new_y)))
                new_x = float(max(self.boundary_margin_inner, min(self.w - self.boundary_margin_outer - 0.5, new_x)))
                self.food[0] = new_y
                self.food[1] = new_x
                self.food_history.append(self.food.detach().cpu().numpy().copy()) # pylint:disable=not-callable

        # Create target position towards food
        with torch.no_grad():
            direction_to_food = self.food - self.position
            dist_to_food = torch.norm(direction_to_food)

            if dist_to_food > self.epsilon:
                direction_to_food = direction_to_food / dist_to_food

            self.target_ = self.position + direction_to_food * self.direction_step_size

    def get_loss(self):
        # Movement loss - move towards target direction
        assert self.target_ is not None
        move_loss = F.mse_loss(self.position, self.target_)

        # Obstacle collision loss (circular obstacles)
        obstacle_loss = 0.0
        for i in range(len(self.obstacles)):
            obs_y, obs_x, obs_r = self.obstacles[i]
            dist_to_obs = torch.norm(self.position - torch.tensor([obs_y, obs_x], dtype=torch.float32, device=self.position.device))
            # Smooth penalty for getting close to obstacle
            dist = dist_to_obs - obs_r - self.obstacle_margin
            obstacle_loss += F.relu(-dist) ** 2 * self.obstacle_penalty_multiplier
        self.obstacle_loss_ = obstacle_loss

        # Boundary loss (stay within map)
        oob = (F.relu(self.position[0] - self.boundary_loss_offset) + F.relu(self.position[0].neg() + self.h - self.boundary_margin_outer) +
               F.relu(self.position[1] - self.boundary_loss_offset) + F.relu(self.position[1].neg() + self.w - self.boundary_margin_outer))
        boundary_loss = oob * self.boundary_penalty_multiplier

        # Self-collision loss
        self_collision_loss = 0.0
        for body_pos in self.body_history[:-1]:  # Exclude the head
            body_tensor = torch.tensor(body_pos, dtype=torch.float32, device=self.position.device)
            dist = torch.norm(self.position - body_tensor)
            self_collision_loss += F.relu(self.self_collision_threshold - dist) ** 2
        self.self_collision_loss_ = self_collision_loss

        # Food attraction (pull towards food)
        self.dist_to_food_ = torch.norm(self.position - self.food)
        food_attraction = self.dist_to_food_

        # Total loss
        total_loss = (
            move_loss +
            self.obstacle_loss_ +
            boundary_loss +
            self.self_collision_loss_ +
            food_attraction
        )

        # Log metrics
        self.log("food collected", self.collected_food.float())
        self.log("obstacle collision", self.obstacle_loss_)
        self.log("self collision", self.self_collision_loss_)
        self.log("distance to food", self.dist_to_food_)

        # Render
        if self._make_images:
            self.log_image(name='snake', image=self._render_frame(), to_uint8=False)

        return total_loss

    def _render_background(self):
        H, W = self.h, self.w
        img = np.full((H * self.cell_size, W * self.cell_size, 3), 30, dtype=np.uint8)

        # Draw checkerboard floor
        for y in range(H):
            for x in range(W):
                color = (40, 40, 50) if (x + y) % 2 == 0 else (45, 45, 55)
                cv2.rectangle(img, (x*self.cell_size, y*self.cell_size),
                              ((x+1)*self.cell_size, (y+1)*self.cell_size), color, -1)

        # Draw circular obstacles
        for obs in self.obstacles:
            obs_y, obs_x, obs_r = obs.cpu().numpy()
            center = (int((obs_x + 0.5) * self.cell_size),
                      int((obs_y + 0.5) * self.cell_size))
            radius = int(obs_r * self.cell_size)
            cv2.circle(img, center, radius, (80, 50, 50), -1)
            cv2.circle(img, center, radius, (150, 80, 80), 2)

        # Draw arena border
        cv2.rectangle(img, (0, 0), (W*self.cell_size-1, H*self.cell_size-1), (100, 100, 120), 3)

        return img

    def _render_frame(self):
        img = self.bg_image.copy()

        # Draw food
        food_pos = self.food.cpu().numpy()
        food_center = (int((food_pos[1] + 0.5) * self.cell_size),
                       int((food_pos[0] + 0.5) * self.cell_size))
        cv2.circle(img, food_center, self.cell_size//3, (0, 255, 0), -1)
        cv2.circle(img, food_center, self.cell_size//3, (255, 255, 255), 1)

        # Draw snake body
        for i, body_pos in enumerate(self.body_history):
            center = (int((body_pos[1] + 0.5) * self.cell_size),
                      int((body_pos[0] + 0.5) * self.cell_size))
            # Gradient from head (brighter) to tail (darker)
            intensity = int(255 * (1 - i / max(len(self.body_history), 1)))
            color = (0, intensity, intensity)
            radius = max(self.cell_size//4, int(self.cell_size//3 * (1 - i / max(len(self.body_history), 1))))
            cv2.circle(img, center, radius, color, -1)

        # Draw snake head
        head_pos = self.position.detach().cpu().numpy() # pylint:disable=not-callable
        head_center = (int((head_pos[1] + 0.5) * self.cell_size),
                       int((head_pos[0] + 0.5) * self.cell_size))
        cv2.circle(img, head_center, self.cell_size//3, (0, 0, 255), -1)
        cv2.circle(img, head_center, self.cell_size//3, (255, 255, 255), 1)

        # Draw score
        score_text = f"Food: {self.collected_food.item()}/{self.target_length}"
        cv2.putText(img, score_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return img
