"""
Same as `python script/run.py ...`, but a fine-tuning run picks up from the newest
checkpoint in its logdir if one exists (so a Colab disconnect only loses the iterations
since the last save). Run from the DPPO repo root.

Note: optimizer state and the running reward scaler are not stored in DPPO checkpoints,
so they restart after a resume; the policy and critic weights continue exactly.
"""

import glob
import os
import re
import runpy
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "script"))  # for `from download_url import ...` in run.py
sys.path.insert(0, REPO)

import agent.finetune.train_agent as train_agent  # noqa: E402

_original_init = train_agent.TrainAgent.__init__


def _init_with_resume(self, cfg):
    _original_init(self, cfg)
    found = []
    for path in glob.glob(os.path.join(self.checkpoint_dir, "state_*.pt")):
        m = re.search(r"state_(\d+)\.pt$", path)
        if m:
            found.append(int(m.group(1)))
    for itr in sorted(found, reverse=True):
        try:
            self.load(itr)
        except Exception as e:  # half-written file from a crash
            bad = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
            os.replace(bad, bad + ".corrupt")
            print(f"[resume] state_{itr}.pt could not be loaded ({e}); trying an older one", flush=True)
            continue
        self.itr = itr + 1
        print(f"[resume] loaded state_{itr}.pt, continuing at iteration {self.itr}", flush=True)
        return
    print("[resume] no checkpoint found, starting from iteration 0", flush=True)


train_agent.TrainAgent.__init__ = _init_with_resume

sys.argv[0] = os.path.join(REPO, "script", "run.py")
runpy.run_path(sys.argv[0], run_name="__main__")
