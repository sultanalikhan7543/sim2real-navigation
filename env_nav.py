"""
Lightweight 2D mobile robot navigation environment with goal occlusion.
Pure NumPy, runs on CPU. Includes two domain randomization profiles:
  - mild DR (training distribution)
  - severe DR (unseen distribution shift for robustness evaluation)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NavigationEnv(gym.Env):
    """
    Differential-drive robot navigating to a goal while avoiding
    static and moving circular obstacles.

    Goal info (dx, dy, dist) is OCCLUDED (zeros) when an obstacle blocks
    line of sight. The policy must remember the last visible goal direction.

    Raw observation (float32):
        [x, y, cos(theta), sin(theta),
         dx_goal, dy_goal, dist_goal,
         lidar_0 ... lidar_{N-1}]
    Action: [linear_vel, angular_vel] in [-1, 1]
    """

    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(
        self,
        world_size=8.0,
        max_steps=250,
        lidar_beams=16,
        lidar_range=4.0,
        goal_threshold=0.5,
        collision_threshold=0.25,
        num_obstacles=5,
        domain_randomization=False,
        severe_dr=False,
    ):
        super().__init__()

        self.world_size = world_size
        self.max_steps = max_steps
        self.lidar_beams = lidar_beams
        self.lidar_range = lidar_range
        self.goal_threshold = goal_threshold
        self.collision_threshold = collision_threshold
        self.num_obstacles = num_obstacles
        self.domain_randomization = domain_randomization
        self.severe_dr = severe_dr

        self.obstacle_radius = 0.4

        raw_obs_dim = 7 + lidar_beams
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(raw_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_theta = 0.0
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.obstacles = np.zeros((num_obstacles, 4), dtype=np.float32)

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_dist = 0.0

        self.dr = {
            "lidar_noise": 0.0,
            "action_latency": 0,
            "goal_noise": 0.0,
            "max_lin_vel": 1.0,
            "max_ang_vel": 1.5,
            "obstacle_speed": 0.5,
        }
        self.action_queue = []

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        half = self.world_size / 2.0 - 1.0

        self.robot_pos = self.np_random.uniform(-half, half, size=2).astype(np.float32)
        self.robot_theta = float(self.np_random.uniform(-np.pi, np.pi))

        for _ in range(100):
            self.goal_pos = self.np_random.uniform(-half, half, size=2).astype(np.float32)
            if np.linalg.norm(self.goal_pos - self.robot_pos) > 2.5:
                break

        self.obstacles = np.zeros((self.num_obstacles, 4), dtype=np.float32)
        for i in range(self.num_obstacles):
            for _ in range(100):
                pos = self.np_random.uniform(-half, half, size=2)
                if (np.linalg.norm(pos - self.robot_pos) > 1.2 and
                        np.linalg.norm(pos - self.goal_pos) > 1.2):
                    break
            self.obstacles[i, :2] = pos

        # Domain randomization profiles
        if self.domain_randomization or self.severe_dr:
            if self.severe_dr:
                # Severe distribution shift: much harder than training
                self.dr["lidar_noise"] = float(self.np_random.uniform(0.08, 0.18))
                self.dr["action_latency"] = int(self.np_random.integers(4, 9))
                self.dr["goal_noise"] = float(self.np_random.uniform(0.15, 0.35))
                self.dr["max_lin_vel"] = float(self.np_random.uniform(0.5, 1.0))
                self.dr["max_ang_vel"] = float(self.np_random.uniform(0.8, 1.5))
                self.dr["obstacle_speed"] = float(self.np_random.uniform(0.8, 1.5))
            else:
                # Mild DR (matches training distribution)
                self.dr["lidar_noise"] = float(self.np_random.uniform(0.0, 0.08))
                self.dr["action_latency"] = int(self.np_random.integers(0, 4))
                self.dr["goal_noise"] = float(self.np_random.uniform(0.0, 0.15))
                self.dr["max_lin_vel"] = float(self.np_random.uniform(0.7, 1.2))
                self.dr["max_ang_vel"] = float(self.np_random.uniform(1.0, 2.0))
                self.dr["obstacle_speed"] = float(self.np_random.uniform(0.0, 0.4))

            for i in range(self.num_obstacles):
                a = self.np_random.uniform(-np.pi, np.pi)
                s = self.np_random.uniform(0.0, self.dr["obstacle_speed"])
                self.obstacles[i, 2] = s * np.cos(a)
                self.obstacles[i, 3] = s * np.sin(a)
        else:
            self.dr["lidar_noise"] = 0.0
            self.dr["action_latency"] = 0
            self.dr["goal_noise"] = 0.0
            self.dr["max_lin_vel"] = 1.0
            self.dr["max_ang_vel"] = 1.5
            self.dr["obstacle_speed"] = 0.0

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.action_queue = []

        obs = self._get_obs()
        self.prev_dist = self._true_dist()
        return obs, {}

    # ------------------------------------------------------------------
    def _true_goal_info(self):
        dx = float(self.goal_pos[0] - self.robot_pos[0])
        dy = float(self.goal_pos[1] - self.robot_pos[1])
        dist = float(np.sqrt(dx * dx + dy * dy))
        return dx, dy, dist

    def _true_dist(self):
        return self._true_goal_info()[2]

    def _line_of_sight_blocked(self):
        p = self.robot_pos
        g = self.goal_pos
        d = g - p
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return False
        d = d / L
        r = self.obstacle_radius
        for i in range(self.num_obstacles):
            o = self.obstacles[i, :2]
            oc = o - p
            t = float(np.dot(oc, d))
            if t < 0.0 or t > L:
                continue
            perp = float(np.linalg.norm(oc - t * d))
            if perp < r:
                return True
        return False

    # ------------------------------------------------------------------
    def _get_obs(self):
        state = np.array(
            [self.robot_pos[0], self.robot_pos[1],
             np.cos(self.robot_theta), np.sin(self.robot_theta)],
            dtype=np.float32,
        )

        dx, dy, dist = self._true_goal_info()

        occluded = self._line_of_sight_blocked()
        if occluded:
            goal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            if (self.domain_randomization or self.severe_dr) and self.dr["goal_noise"] > 0.0:
                dx += float(self.np_random.normal(0.0, self.dr["goal_noise"]))
                dy += float(self.np_random.normal(0.0, self.dr["goal_noise"]))
                dist = float(np.sqrt(dx * dx + dy * dy))
            goal = np.array([dx, dy, dist], dtype=np.float32)

        lidar = self._raycast()
        if (self.domain_randomization or self.severe_dr) and self.dr["lidar_noise"] > 0.0:
            lidar = lidar + self.np_random.normal(
                0.0, self.dr["lidar_noise"], size=lidar.shape
            ).astype(np.float32)
            lidar = np.clip(lidar, 0.0, self.lidar_range).astype(np.float32)

        return np.concatenate([state, goal, lidar]).astype(np.float32)

    # ------------------------------------------------------------------
    def _raycast(self):
        angles = np.linspace(-np.pi, np.pi, self.lidar_beams, endpoint=False)
        angles = angles + self.robot_theta
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

        half = self.world_size / 2.0
        eps = 1e-8
        px, py = float(self.robot_pos[0]), float(self.robot_pos[1])

        dx_ = dirs[:, 0]
        dy_ = dirs[:, 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(
                dx_ > eps, (half - px) / (dx_ + eps),
                np.where(dx_ < -eps, (-half - px) / (dx_ - eps), np.inf),
            )
            ty = np.where(
                dy_ > eps, (half - py) / (dy_ + eps),
                np.where(dy_ < -eps, (-half - py) / (dy_ - eps), np.inf),
            )

        tx = np.where(tx > 0, tx, np.inf)
        ty = np.where(ty > 0, ty, np.inf)
        boundary_t = np.minimum(tx, ty).astype(np.float32)

        dists = np.full(self.lidar_beams, self.lidar_range, dtype=np.float32)
        dists = np.minimum(dists, boundary_t)

        r = self.obstacle_radius
        r2 = r * r

        for i in range(self.num_obstacles):
            ox = float(self.obstacles[i, 0])
            oy = float(self.obstacles[i, 1])
            ocx = ox - px
            ocy = oy - py
            t_proj = dirs[:, 0] * ocx + dirs[:, 1] * ocy
            oc_sq = ocx * ocx + ocy * ocy
            perp_sq = oc_sq - t_proj * t_proj
            hit = (t_proj > 0) & (perp_sq < r2) & (perp_sq >= 0)
            if np.any(hit):
                offset = np.sqrt(np.maximum(r2 - perp_sq, 0.0))
                t_hit = t_proj - offset
                t_hit = np.where(hit & (t_hit > 0), t_hit, np.inf)
                dists = np.minimum(dists, t_hit.astype(np.float32))

        return np.clip(dists, 0.0, self.lidar_range).astype(np.float32)

    # ------------------------------------------------------------------
    def _apply_dynamics(self, action):
        lin = float(np.clip(action[0], -1.0, 1.0)) * self.dr["max_lin_vel"]
        ang = float(np.clip(action[1], -1.0, 1.0)) * self.dr["max_ang_vel"]
        dt = 0.1

        self.robot_pos[0] += lin * np.cos(self.robot_theta) * dt
        self.robot_pos[1] += lin * np.sin(self.robot_theta) * dt
        self.robot_theta = float((self.robot_theta + ang * dt + np.pi) % (2 * np.pi) - np.pi)

    def _move_obstacles(self):
        half = self.world_size / 2.0 - self.obstacle_radius - 0.1
        dt = 0.1
        self.obstacles[:, 0] += self.obstacles[:, 2] * dt
        self.obstacles[:, 1] += self.obstacles[:, 3] * dt
        for i in range(self.num_obstacles):
            if abs(self.obstacles[i, 0]) > half:
                self.obstacles[i, 2] *= -1.0
                self.obstacles[i, 0] = float(np.clip(self.obstacles[i, 0], -half, half))
            if abs(self.obstacles[i, 1]) > half:
                self.obstacles[i, 3] *= -1.0
                self.obstacles[i, 1] = float(np.clip(self.obstacles[i, 1], -half, half))

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).flatten()[:2]

        if (self.domain_randomization or self.severe_dr) and self.dr["action_latency"] > 0:
            self.action_queue.append(action.copy())
            if len(self.action_queue) <= self.dr["action_latency"]:
                applied = self.prev_action
            else:
                applied = self.action_queue.pop(0)
        else:
            applied = action

        self._apply_dynamics(applied)
        self._move_obstacles()

        obs = self._get_obs()

        _, _, dist = self._true_goal_info()
        goal_reached = dist < self.goal_threshold

        if self.num_obstacles > 0:
            obs_d = np.linalg.norm(self.obstacles[:, :2] - self.robot_pos, axis=1)
            min_obs_dist = float(np.min(obs_d))
        else:
            min_obs_dist = np.inf
        collision = min_obs_dist < (self.obstacle_radius + self.collision_threshold)

        half = self.world_size / 2.0
        if abs(self.robot_pos[0]) > half or abs(self.robot_pos[1]) > half:
            collision = True

        progress = (self.prev_dist - dist) * 15.0
        r_goal = 200.0 if goal_reached else 0.0
        r_collision = -50.0 if collision else 0.0
        r_time = -0.02
        r_smooth = -0.1 * float(np.linalg.norm(action - self.prev_action))
        lidar = obs[7:]
        r_prox = -0.01 * float(np.sum(lidar < 0.3))

        reward = float(progress + r_goal + r_collision + r_time + r_smooth + r_prox)
        if dist < 1.0 and not collision:
            reward += 0.5 * (1.0 - dist)

        terminated = bool(goal_reached or collision)
        truncated = bool(self.step_count >= self.max_steps)

        self.prev_action = action.copy()
        self.prev_dist = dist

        info = {
            "is_success": bool(goal_reached),
            "is_collision": bool(collision),
            "distance_to_goal": dist,
            "step_count": self.step_count,
            "min_obstacle_dist": min_obs_dist,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


# ======================================================================
# HistoryWrapper: stacks the last H observations into a single vector.
# ======================================================================
class HistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, history_len=5):
        super().__init__(env)
        self.history_len = history_len
        self.base_obs_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(history_len * self.base_obs_dim,),
            dtype=np.float32,
        )
        self.history = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history = np.tile(obs, (self.history_len, 1)).astype(np.float32)
        return self.history.flatten().astype(np.float32), info

    def observation(self, obs):
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = obs.astype(np.float32)
        return self.history.flatten().astype(np.float32)