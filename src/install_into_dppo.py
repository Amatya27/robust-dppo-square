"""
Copies the files in src/ into a DPPO checkout and registers the perturbation wrapper.

    python src/install_into_dppo.py /path/to/dppo

Safe to run more than once. The only upstream file touched is
env/gym_utils/wrapper/__init__.py, where three lines are appended.
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

REGISTRATION = (
    "\n# perturbation-aware wrapper (all perturbations off by default)\n"
    "from .perturb import PerturbedRobomimicLowdimWrapper\n"
    'wrapper_dict["robomimic_lowdim"] = PerturbedRobomimicLowdimWrapper\n'
)


def main(repo):
    init = os.path.join(repo, "env", "gym_utils", "wrapper", "__init__.py")
    if not os.path.exists(init):
        sys.exit(f"{repo} does not look like a DPPO checkout (missing {init})")

    shutil.copy(os.path.join(HERE, "perturb.py"), os.path.join(repo, "env", "gym_utils", "wrapper", "perturb.py"))
    for name in ("dppo_run_resumable.py", "perturb_selftest.py"):
        shutil.copy(os.path.join(HERE, name), os.path.join(repo, name))

    with open(init) as f:
        registered = "PerturbedRobomimicLowdimWrapper" in f.read()
    if not registered:
        with open(init, "a") as f:
            f.write(REGISTRATION)
    print("installed into", repo, "(already registered)" if registered else "")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
