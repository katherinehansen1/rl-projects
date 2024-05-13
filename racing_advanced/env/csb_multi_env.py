"""
Original game: https://www.codingame.com/multiplayer/bot-programming/coders-strike-back
"""
import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import os

class Vec:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, (int, float)):  # Multiplication by a scalar
            return Vec(self.x * other, self.y * other)
        raise NotImplementedError("Can only multiply Vec by a scalar")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):  # Division by a scalar
            return Vec(self.x / other, self.y / other)
        raise NotImplementedError("Can only divide Vec by a scalar")

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def to_tuple(self):
        return (self.x, self.y)

    def round(self):
        self.x = round(self.x)
        self.y = round(self.y)
        return self

    def int(self):
        self.x = int(self.x)
        self.y = int(self.y)
        return self

    def sqr_norm(self):
        return self.x**2 + self.y**2

    def normalize(self):
        norm = np.sqrt(self.sqr_norm())
        if norm == 0:
            return Vec(0, 0)  # prevent division by zero
        return Vec(self.x / norm, self.y / norm)

    def perpendicular(self):
        return Vec(-self.y, self.x)

def get_sqr_distance(v1,v2):
    x_diff = v1.x - v2.x
    y_diff = v1.y - v2.y
    return x_diff**2 + y_diff**2

def get_distance(v1, v2):
    return math.sqrt(get_sqr_distance(v1,v2))

def get_angle(v1, v2):
    x_diff = v2.x - v1.x
    y_diff = v2.y - v1.y
    return math.atan2(y_diff, x_diff)

def normalize_angle(angle):
    if angle > math.pi:
        angle -= 2 * math.pi  # Turn left
    elif angle < -math.pi:
        angle += 2 * math.pi  # Turn right
    return angle

def angle_to_vector(angle):
    return Vec(math.cos(angle), math.sin(angle))

def intersects_moving_circles(p1, v1, p2, v2, r1, r2):
    """
    Checks if two moving circles defined by their initial positions (p1, p2),
    velocities (v1, v2), and radii (r1, r2) intersect.
    """
    d = v1 - v2  # Relative velocity
    f = p1 - p2  # Relative position
    r = r1 + r2  # Sum of radii

    a = d.dot(d)
    b = 2 * f.dot(d)
    c = f.dot(f) - r * r

    if a == 0:  # The velocities are the same or both are stationary
        if f.sqr_norm() <= r**2:
            return True, 0  # Starting within collision range
        else:
            return False, None

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        # No intersection
        return False, None

    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    # Check for valid time within the time step [0, 1]
    valid_times = [t for t in (t1, t2) if 0 <= t <= 1]
    if valid_times:
        return True, min(valid_times)  # Return the earliest valid collision time
    else:
        return False, None

def intersects_circle(p_prev, p_curr, center, radius):
    d = p_curr - p_prev
    f = p_prev - center
    a = d.dot(d)
    b = 2 * f.dot(d)
    c = f.dot(f) - radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0 or a==0:
        # No intersection
        return False, None

    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    if (0 <= t1 <= 1):
        return True, t1
    elif (0 <= t2 <= 1):
        return True, t2
    else:
        return False, None


class Racer:
    def __init__(self, aid, tid):
        self.aid = aid
        self.team_id = tid
        self.max_thrust = 100.0
        self.maxSteeringAngle = 0.1 * np.pi
        self.friction = 0.85

    def move(self, target, thrust, dt):
        self.pos_prev = self.pos
        da = self.get_delta_angle(target)
        da = np.clip(da, -self.maxSteeringAngle*dt, self.maxSteeringAngle*dt)

        self.theta = normalize_angle(self.theta + da)
        self.vel += thrust * angle_to_vector(self.theta) * dt
        self.steps_since_last_checkpoint += dt
        self.pos = (self.pos + self.vel * dt).round()

    def finalize_move(self, dt):
        self.vel = (self.vel * (self.friction**dt)).int()
        if self.steps_since_last_checkpoint >= 100:
            self.late = True

    def get_delta_angle(self, target):
        angle_to_target = get_angle(self.pos, target)
        delta_angle = angle_to_target - self.theta
        return normalize_angle(delta_angle)

    def pass_checkpoint(self, time):
        self.current_checkpoint += 1
        self.steps_since_last_checkpoint = 0
        if self.current_checkpoint >= self.num_checkpoints:
            self.late = False
            self.current_checkpoint = 0
            self.laps_remaining -= 1
            if self.laps_remaining == 0:
                self.done = True
                self.finished = True
                self.finish_time = time

    def reset(self, pos, theta, num_checkpoints):
        self.current_checkpoint = 0
        self.laps_remaining = 3
        self.num_checkpoints = num_checkpoints
        self.steps_since_last_checkpoint = 0

        self.pos = pos
        self.vel = Vec(0, 0)
        self.theta = theta
        self.pos_prev = None

        self.done = False
        self.failed = False
        self.finished = False
        self.finish_time = None
        self.late = False

class CodersStrikeBackMultiBase:
    metadata = {"render.modes": ["human"], "video.frames_per_second": 30}
    def __init__(self, dt=1):
        self.max_checkpoints = 5
        self.gamePixelWidth = 16000.
        self.gamePixelHeight = 9000.
        self.checkpoint_radius = 600
        self.num_racers_per_team = 2
        self.num_teams = 3
        self.racers = []
        self.checkpoints = self.sample_checkpoints(self.max_checkpoints)
        self.viewer = None
        self.n_laps = 3
        self.pod_radius = 400
        self.dt = dt

        self.teams = set(range(self.num_teams))

        # Initialize racers
        for c in range(self.num_racers_per_team):
            for t in range(self.num_teams):
                self.racers.append(Racer(f'car_{t}_{c}', t))

    def step(self, targets, thrusts):
        if self.race_over():
            return

        for racer in self.racers:
            racer.move(targets[racer.aid], thrusts[racer.aid], self.dt)

        self.resolve_all_collisions()
        self.check_checkpoints()

        for racer in self.racers:
            racer.finalize_move(self.dt)

        self.fail_late_teams()
        self.time += self.dt

    def fail_late_teams(self):
        failed = {team: True for team in self.teams}
        for racer in self.racers:
            if not racer.late:
                failed[racer.team_id] = False
        for racer in self.racers:
            if failed[racer.team_id]:
                racer.failed = True
                racer.done = True


    def resolve_all_collisions(self):
        for i in range(len(self.racers)):
            for j in range(i + 1, len(self.racers)):
                r1 = self.racers[i]
                r2 = self.racers[j]
                collided, collision_time = intersects_moving_circles(r1.pos_prev, r1.vel, r2.pos_prev, r2.vel, self.pod_radius, self.pod_radius)
                if collided:
                    self.resolve_collision_at_time(self.racers[i], self.racers[j], collision_time, self.dt)

    def check_checkpoints(self):
        for racer in self.racers:

            passed_check, t = intersects_circle(racer.pos_prev, racer.pos, self.checkpoints[racer.current_checkpoint], self.checkpoint_radius)
            if passed_check:
                racer.pass_checkpoint(self.time + t)

    def resolve_collision_at_time(self, racer1, racer2, t, dt):
        # Move each racer to the point of collision
        initial_pos1 = racer1.pos_prev
        initial_pos2 = racer2.pos_prev
        racer1.pos = initial_pos1 + (racer1.vel * dt*t)
        racer2.pos = initial_pos2 + (racer2.vel * dt*t)

        # Calculate new velocities post-collision
        self.handle_collision(racer1, racer2)

        # Move racers for the remaining time step t_remain
        t_remain = 1 - t
        racer1.pos += racer1.vel * t_remain * dt
        racer2.pos += racer2.vel * t_remain * dt


    def handle_collision(self, racer1, racer2):
        # Normal vector between the two colliding pods
        normal = racer1.pos - racer2.pos
        normal = normal.normalize()

        # Relative velocity vector
        rel_vel = racer1.vel - racer2.vel
        velocity_along_normal = rel_vel.dot(normal)

        # Ensure the collision is valid (pods are moving towards each other)
        if velocity_along_normal > 0:
            return

        # Calculate impulse magnitude
        restitution = 0.5  # Coefficient of restitution for elastic collision
        impulse_magnitude = -(1 + restitution) * velocity_along_normal
        impulse_magnitude = max(impulse_magnitude, 120)  # Minimum impulse

        # Apply impulse to both racers
        impulse = normal * impulse_magnitude
        racer1.vel += impulse
        racer2.vel -= impulse


    def sample_checkpoints(self, n):
        checkpoints = []
        while len(checkpoints) < n:
            x = np.random.randint(self.checkpoint_radius, self.gamePixelWidth - self.checkpoint_radius)
            y = np.random.randint(self.checkpoint_radius, self.gamePixelHeight - self.checkpoint_radius)
            checkpoint = Vec(x, y)
            if all(get_sqr_distance(checkpoint, cp) > (3*self.checkpoint_radius)**2 for cp in checkpoints):
                checkpoints.append(checkpoint)
        return checkpoints


    def race_over(self):
        return any(racer.done for racer in self.racers)

    def winning_teams(self):
        if not self.race_over():
            return set()
        failing_teams = set()
        winner = None
        winning_time = None
        for racer in self.racers:
            if racer.done:
                if racer.failed:
                    failing_teams.add(racer.team_id)
                if racer.finished:
                    if winner is None or racer.finish_time < winning_time:
                        winner = racer.team_id
                        winning_time = racer.finish_time
        if winner:
            return set([winner])
        else:
            return self.teams.difference(failing_teams)


    def reset(self):
        self.n_checkpoints = np.random.randint(3,self.max_checkpoints + 1)
        self.checkpoints = self.sample_checkpoints(self.n_checkpoints)
        self.time = 0
        self.viewer = None

        start_checkpoint = self.checkpoints[-1]
        end_checkpoint = self.checkpoints[0]

        direction_vector = end_checkpoint - start_checkpoint
        perpendicular = direction_vector.perpendicular().normalize()

        num_racers = len(self.racers)
        midpoint_index = (num_racers-1) / 2
        spacing = 3 * self.pod_radius

        # Initialize racers
        for i, racer in enumerate(self.racers):
            offset_index = i - midpoint_index
            offset_distance = offset_index * spacing
            start_pos = start_checkpoint + perpendicular * offset_distance
            theta = get_angle(start_pos, end_checkpoint)
            racer.reset(start_pos, theta, self.n_checkpoints)

    def get_targets(self):
        racer_targets = {}
        for racer in self.racers:
            targets = []
            n = self.n_checkpoints
            cur_ind = racer.current_checkpoint
            for i in range(self.max_checkpoints):
                targets.append(self.checkpoints[(i+cur_ind) % n])
            racer_targets[racer.aid] = targets

        return racer_targets



    def render(self, mode="human"):
        # Must be 16:9
        screen_width = 640
        screen_height = 360
        scale = screen_width / self.gamePixelWidth
        pod_diam = scale * self.pod_radius * 2.0
        checkpoint_diam = scale * self.checkpoint_radius * 2.0

        if self.viewer is None:
            import pygame_rendering
            self.viewer = pygame_rendering.Viewer(screen_width, screen_height)

            dirname = os.path.dirname(__file__)
            backImgPath = os.path.join(dirname, "imgs", "back.png")
            self.viewer.setBackground(backImgPath)

            ckptImgPath = backImgPath = os.path.join(dirname, "imgs", "ckpt.png")

            self.checkpointCircle = []
            for i in range(self.n_checkpoints):
                if i == self.n_checkpoints - 1:
                    display_num = "End"
                else:
                    display_num = i+1
                ckpt = scale * self.checkpoints[i]
                ckptObject = pygame_rendering.Checkpoint(
                    ckptImgPath,
                    pos=ckpt.to_tuple(),
                    number=display_num,
                    width=checkpoint_diam,
                    height=checkpoint_diam,
                )
                ckptObject.setVisible(True)
                self.viewer.addCheckpoint(ckptObject)

            podImgPaths = backImgPaths = [os.path.join(dirname, "imgs", "pod.png"),
                                          os.path.join(dirname, "imgs", "pod2.png")]
            for racer in self.racers:
                pod = scale * racer.pos
                podObject = pygame_rendering.Pod(
                    podImgPaths[racer.team_id],
                    pos=pod.to_tuple(),
                    theta=racer.theta,
                    width=pod_diam,
                    height=pod_diam,
                )
                self.viewer.addPod(podObject)

            text = pygame_rendering.Text(
                "Time", backgroundColor=(0, 0, 0), pos=(0, 0)
            )
            self.viewer.addText(text)

        for i, racer in enumerate(self.racers):
            self.viewer.pods[i].setPos((scale*racer.pos).to_tuple())
            self.viewer.pods[i].rotate(racer.theta)


        remaining_laps = min(racer.laps_remaining for racer in self.racers)
        self.viewer.text.setText(f'Time: {self.time}  Lap: {1+self.n_laps - remaining_laps}/{self.n_laps}')
        return self.viewer.render()


    def close(self):
        if self.viewer:
            self.viewer.close()

def make_agent_dictionary(agent_ids, res):
    return {agent_id: res for agent_id in agent_ids}

class CodersStrikeBackMulti(CodersStrikeBackMultiBase, MultiAgentEnv):
    def __init__(self, seed=None, dt=1):
        MultiAgentEnv.__init__(self)
        CodersStrikeBackMultiBase.__init__(self, dt)

        min_pos = -200000.0
        max_pos = 200000.0
        min_vel = -2000.0
        max_vel = 2000.0
        screen_max = [self.gamePixelWidth, self.gamePixelHeight]
        self.ind_observation_space = spaces.Box(
            low=np.array([0, -np.pi, min_pos, min_pos, min_vel, min_vel]+[0,0]*self.max_checkpoints),
            high=np.array([self.n_laps, np.pi, max_pos, max_pos, max_vel, max_vel]+screen_max*self.max_checkpoints),
            dtype=np.float64
        )

        self.ind_action_space = spaces.Box(
            low = np.array([min_pos, min_pos, 0.0]),
            high = np.array([max_pos, max_pos, self.racers[0].max_thrust]),
            dtype=np.float64
        )

        self._agent_ids = [racer.aid for racer in self.racers]
        self.action_space = gym.spaces.Dict(make_agent_dictionary(self._agent_ids, self.ind_action_space))
        self.observation_space = gym.spaces.Dict(make_agent_dictionary(self._agent_ids, self.ind_observation_space))

    def get_observations(self):
        targets = self.get_targets()
        all_obs = {}
        for racer in self.racers:
            obs = [racer.laps_remaining, racer.theta,
                   racer.pos.x, racer.pos.y,
                   racer.vel.x, racer.vel.y,
                   ]
            for t in targets[racer.aid]:
                obs += [t.x, t.y]
            all_obs[racer.aid] = np.array(obs)
        return all_obs

    def reset(self, seed=None, options=None):
        super().reset()
        return self.get_observations(), {}

    def step(self, actions):
        targets = {aid: Vec(action[0], action[1]) for aid, action in actions.items()}
        thrusts = {aid: action[2] for aid, action in actions.items()}
        super().step(targets, thrusts)
        done = self.race_over()
        dones = {r.aid: done for r in self.racers}
        dones["__all__"] = done
        return self.get_observations(), self.reward(), dones, dones, {}

    def render(self):
        return super().render()

    def close(self):
        if self.viewer:
            self.viewer.close()

    def evaluation_reward(self):
        if not self.race_over():
            return make_agent_dictionary(self._agent_ids, 0)
        rewards = {}
        winners = self.winning_teams()
        for racer in self.racers:
            if racer.team_id in winners:
                rewards[racer.aid] = 1
            else:
                rewards[racer.aid] = -1
        return rewards

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self):
        return self.evaluation_reward()
