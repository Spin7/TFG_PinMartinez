import math
from enum import Enum


# ==============================
# STATES
# ==============================

class State(Enum):
    JUVENILE = 0
    ADULT_MALE = 1
    ADULT_FEMALE_U = 2
    ADULT_FEMALE_G = 3
    DEAD = 4


# ==============================
# Utility
# ==============================

def rate_to_prob(rate, dt):
    return 1.0 - math.exp(-rate * dt)


# ==============================
# Agent
# ==============================

class Agent:

    def __init__(self, state, pos, rng):

        self.state = state
        self.pos = (int(pos[0]), int(pos[1]))
        self.rng = rng

        self.prev_theta = rng.uniform(0, 2 * math.pi)

    # ==================================================
    # DEMOGRAPHIC DYNAMICS
    # ==================================================

    def step(self, dt, env):

        if self.state == State.DEAD:
            return

        if self.state == State.JUVENILE:
            self._juvenile_step(dt, env)

        elif self.state == State.ADULT_MALE:
            self._adult_male_step(dt, env)

        elif self.state == State.ADULT_FEMALE_U:
            self._female_unmated_step(dt, env)

        elif self.state == State.ADULT_FEMALE_G:
            self._female_mated_step(dt, env)

    # --------------------------------------------------

    def _juvenile_step(self, dt, env):

        mu_J = env.mu_J

        nb = env.neighborhood_counts(self.pos, env.density_radius)
        J_local = nb["J"]

        area = (2 * env.density_radius + 1) ** 2
        density_term = env.alpha * J_local / (env.Kc * area)

        mu_total = mu_J + density_term

        if self.rng.random() < rate_to_prob(mu_total, dt):
            self.state = State.DEAD
            return

        # maturation
        if self.rng.random() < rate_to_prob(env.gamma, dt):

            if self.rng.random() < 0.5:
                self.state = State.ADULT_FEMALE_U
            else:
                self.state = State.ADULT_MALE

    # --------------------------------------------------

    def _adult_male_step(self, dt, env):

        if self.rng.random() < rate_to_prob(env.mu_M, dt):
            self.state = State.DEAD

    # --------------------------------------------------

    def _female_unmated_step(self, dt, env):

        if self.rng.random() < rate_to_prob(env.mu_F, dt):
            self.state = State.DEAD
            return

        nb = env.neighborhood_counts(self.pos, env.mating_radius)
        M_local = nb["M"]

        # Per-male encounter rate: beta is encounters/day per male in radius.
        # No area normalization — dividing by a large area made mating
        # nearly impossible at realistic (sparse) population densities.
        lambda_m = env.beta * M_local

        if self.rng.random() < rate_to_prob(lambda_m, dt):
            self.state = State.ADULT_FEMALE_G

    # --------------------------------------------------

    def _female_mated_step(self, dt, env):

        if self.rng.random() < rate_to_prob(env.mu_F, dt):
            self.state = State.DEAD
            return

        if self.rng.random() < rate_to_prob(env.f, dt):
            env.register_new_juvenile(self.pos)

    # ==================================================
    # MOVEMENT
    # ==================================================

    def move(self, dt, env):

        if self.state == State.DEAD:
            return

        if self.state == State.JUVENILE:
            return

        # Discrete lattice movement (paper model):
        # sample next cell from N_R(x) weighted by potential + distance.
        self.pos = env.sample_movement(self.pos, self.rng)