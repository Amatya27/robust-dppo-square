"""
Checks that the perturbations do what they are supposed to do. Run from the DPPO repo root:
    python perturb_selftest.py <normalization.npz> <out.json>
"""

import json
import sys

import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from env.gym_utils.wrapper.robomimic_lowdim import RobomimicLowdimWrapper
from env.gym_utils.wrapper.perturb import PerturbedRobomimicLowdimWrapper
from env.gym_utils.wrapper import wrapper_dict

NORM, OUT = sys.argv[1], sys.argv[2]
KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": KEYS})
with open("cfg/robomimic/env_meta/square.json") as f:
    META = json.load(f)
META["reward_shaping"] = False


def make(cls=PerturbedRobomimicLowdimWrapper, **kw):
    env = EnvUtils.create_env_from_metadata(env_meta=META, render=False, render_offscreen=False, use_image_obs=False)
    env.env.hard_reset = False
    return cls(env, normalization_path=NORM, low_dim_keys=KEYS, **kw)


def nut_xy(w):
    rs = w.env.env
    return rs.sim.data.body_xpos[rs.obj_body_id[rs.nuts[rs.nut_id].name]][:2].copy()


checks = {}
checks["registered"] = wrapper_dict["robomimic_lowdim"] is PerturbedRobomimicLowdimWrapper

orig, ours = make(RobomimicLowdimWrapper), make()
orig.seed(7)
o1 = orig.reset()
ours.seed(7)
o2 = ours.reset()
checks["defaults_identical_obs"] = bool(np.allclose(o1["state"], o2["state"]))
checks["defaults_identical_nut_xy"] = bool(np.allclose(nut_xy(orig), nut_xy(ours)))

shifted = make(shift_cm=4.0)
shifted.seed(7); shifted.reset()
d = float(np.linalg.norm(nut_xy(shifted) - nut_xy(orig)) * 100)
checks["shift_4cm_measured_cm"] = round(d, 4)
checks["shift_ok"] = abs(d - 4.0) < 0.05

phys = make(friction_scale=2.0, mass_scale=0.5)
phys.seed(7); phys.reset()
sim, base_sim = phys.env.env.sim, orig.env.env.sim
fr = np.array(sim.model.geom_friction[:, 0]) / np.maximum(np.array(base_sim.model.geom_friction[:, 0]), 1e-9)
b = phys._nut_body
checks["friction_ratio_mean"] = round(float(fr.mean()), 4)
checks["mass_ratio"] = round(float(sim.model.body_mass[b] / base_sim.model.body_mass[b]), 4)
checks["physics_ok"] = abs(checks["friction_ratio_mean"] - 2.0) < 1e-3 and abs(checks["mass_ratio"] - 0.5) < 1e-3

noisy = make(obs_noise=0.05)
noisy.seed(7)
o3 = noisy.reset()
checks["noise_std_measured"] = round(float(np.std(o3["state"] - o1["state"])), 4)

dr = make(dr=True)
dr.seed(3)
draws = []
for _ in range(20):
    dr.reset()
    draws.append(dr.last_perturbation)
checks["dr_shift_range_cm"] = [round(min(x["shift_cm"] for x in draws), 3), round(max(x["shift_cm"] for x in draws), 3)]
checks["dr_friction_range"] = [round(min(x["friction_scale"] for x in draws), 3), round(max(x["friction_scale"] for x in draws), 3)]
checks["dr_ok"] = all(0 <= x["shift_cm"] <= 4.0 and 0.5 <= x["friction_scale"] <= 2.0 and 0.5 <= x["mass_scale"] <= 2.0 for x in draws)

for w in (orig, shifted, phys, noisy, dr):
    for _ in range(20):
        w.step(np.zeros(7))
checks["steps_ok"] = True

checks["ALL_OK"] = bool(
    checks["registered"] and checks["defaults_identical_obs"] and checks["defaults_identical_nut_xy"]
    and checks["shift_ok"] and checks["physics_ok"] and checks["dr_ok"]
)
with open(OUT, "w") as f:
    json.dump(checks, f, indent=2)
print(json.dumps(checks, indent=2))
