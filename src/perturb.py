"""
DPPO's Robomimic low-dim wrapper with optional perturbations, used for our robustness tests.

Everything is off by default, and then it behaves exactly like the original wrapper
(no extra random draws, no physics changes).

Fixed perturbations for testing (same level every episode):
  shift_cm        move the nut this many cm in a random direction after reset
  obs_noise       std of Gaussian noise added to the normalized 23-dim state
  friction_scale  multiply the sliding friction of every geom in the scene
  mass_scale      multiply the mass of the nut

Domain randomization for Robust DPPO (dr=True), new values every episode:
  shift ~ U(0, dr_shift_max_cm), noise std ~ U(0, dr_noise_max),
  friction and mass ~ log-uniform(dr_scale_low, dr_scale_high)
"""

import numpy as np

from .robomimic_lowdim import RobomimicLowdimWrapper


class PerturbedRobomimicLowdimWrapper(RobomimicLowdimWrapper):
    def __init__(
        self,
        env,
        shift_cm=0.0,
        obs_noise=0.0,
        friction_scale=1.0,
        mass_scale=1.0,
        dr=False,
        dr_shift_max_cm=4.0,
        dr_noise_max=0.05,
        dr_scale_low=0.5,
        dr_scale_high=2.0,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.shift_cm = float(shift_cm)
        self.obs_noise = float(obs_noise)
        self.friction_scale = float(friction_scale)
        self.mass_scale = float(mass_scale)
        self.dr = bool(dr)
        self.dr_shift_max_cm = float(dr_shift_max_cm)
        self.dr_noise_max = float(dr_noise_max)
        self.dr_scale_low = float(dr_scale_low)
        self.dr_scale_high = float(dr_scale_high)
        self.active = (
            self.dr
            or self.shift_cm > 0
            or self.obs_noise > 0
            or self.friction_scale != 1.0
            or self.mass_scale != 1.0
        )
        self._rng = None
        self._orig = None
        self._noise_now = 0.0
        self.last_perturbation = {}

    def _robosuite_env(self):
        return self.env.env  # robomimic EnvRobosuite -> robosuite NutAssemblySquare

    def _setup_once(self):
        rs = self._robosuite_env()
        sim = rs.sim
        if self._orig is None:
            nut = rs.nuts[rs.nut_id]
            self._nut_joint = nut.joints[0]
            self._nut_body = rs.obj_body_id[nut.name]
            self._orig = {
                "friction": np.array(sim.model.geom_friction[:, 0]).copy(),
                "mass": float(sim.model.body_mass[self._nut_body]),
            }
        if self._rng is None:
            # seeded from numpy's global state, which DPPO seeds per env (seed + env index)
            self._rng = np.random.RandomState(np.random.randint(2**31 - 1))

    def _draw(self):
        if self.dr:
            shift = self._rng.uniform(0.0, self.dr_shift_max_cm)
            noise = self._rng.uniform(0.0, self.dr_noise_max)
            lo, hi = np.log(self.dr_scale_low), np.log(self.dr_scale_high)
            fric = float(np.exp(self._rng.uniform(lo, hi)))
            mass = float(np.exp(self._rng.uniform(lo, hi)))
        else:
            shift, noise = self.shift_cm, self.obs_noise
            fric, mass = self.friction_scale, self.mass_scale
        angle = self._rng.uniform(0.0, 2.0 * np.pi)
        return shift, angle, noise, fric, mass

    def _apply_physics(self, shift_cm, angle, fric, mass):
        sim = self._robosuite_env().sim
        sim.model.geom_friction[:, 0] = self._orig["friction"] * fric
        sim.model.body_mass[self._nut_body] = self._orig["mass"] * mass
        if shift_cm > 0:
            qpos = np.array(sim.data.get_joint_qpos(self._nut_joint), dtype=float).copy()
            qpos[0] += shift_cm / 100.0 * np.cos(angle)
            qpos[1] += shift_cm / 100.0 * np.sin(angle)
            sim.data.set_joint_qpos(self._nut_joint, qpos)
        sim.forward()

    def _noisy(self, obs):
        if self._noise_now > 0:
            state = obs["state"]
            obs["state"] = state + self._rng.normal(0.0, self._noise_now, size=state.shape).astype(state.dtype)
        return obs

    def reset(self, options={}, **kwargs):
        obs = super().reset(options=options, **kwargs)
        if not self.active:
            return obs
        self._setup_once()
        shift, angle, noise, fric, mass = self._draw()
        self._apply_physics(shift, angle, fric, mass)
        self._noise_now = noise
        self.last_perturbation = dict(shift_cm=shift, angle=angle, obs_noise=noise, friction_scale=fric, mass_scale=mass)
        obs = self.get_observation(self.env.get_observation())  # re-read the state after moving the nut
        return self._noisy(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.active:
            obs = self._noisy(obs)
        return obs, reward, done, info
