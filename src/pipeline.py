"""
Runs all the training and evaluation jobs for the DPPO robustness study, one after another.

    python pipeline.py config.json

Runs the job list in priority order, one job at a time. Finished jobs are detected from
files on Google Drive, so after a disconnect you just start it again and it carries on.
Every minute it writes STATUS_<role>.md and status_<role>.json into <project>/pipeline/.

Control files (create them in <project>/pipeline/ from Drive or the notebook):
    STOP              finish the current job, then exit
    SKIP_<job name>   never run that job
"""

import datetime as dt
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

CFG = json.load(open(sys.argv[1]))
ROLE = CFG["role"]
P = CFG["project"]
REPO = CFG["repo"]
PY = CFG["python"]
T = CFG["train"]
E = CFG["eval"]
PIPE_DIR = os.path.join(P, "pipeline")
os.makedirs(PIPE_DIR, exist_ok=True)
STATUS_JSON = os.path.join(PIPE_DIR, f"status_{ROLE}.json")
STATUS_MD = os.path.join(PIPE_DIR, f"STATUS_{ROLE}.md")
EVENTS = os.path.join(PIPE_DIR, f"events_{ROLE}.log")
EP_LEN = CFG["max_episode_steps"] // CFG["act_steps"]  # policy calls per episode (400 / 4 = 100)

ENV = dict(os.environ)
ENV.update(CFG["env_vars"])

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(CFG.get("timezone", "America/Phoenix"))
except Exception:
    TZ = None


def now():
    return dt.datetime.now(TZ) if TZ else dt.datetime.now()


def stamp(t=None):
    return (t or now()).strftime("%Y-%m-%d %H:%M:%S")


def event(msg):
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    with open(EVENTS, "a") as f:
        f.write(line + "\n")


# Job list
def train_dir(kind, seed):
    return os.path.join(P, "runs", "train", f"{kind}_s{seed}")


def final_ckpt(kind, seed):
    return os.path.join(train_dir(kind, seed), "checkpoint", f"state_{T['n_itr'] - 1}.pt")


def train_job(kind, seed):
    logdir = train_dir(kind, seed)
    cmd = [
        PY, "dppo_run_resumable.py",
        "--config-dir=cfg/robomimic/finetune/square", "--config-name=ft_ppo_diffusion_mlp",
        f"logdir={logdir}", f"seed={seed}", "wandb=null",
        f"base_policy_path={CFG['pretrained_ckpt']}",
        f"env.n_envs={T['n_envs']}", f"train.n_steps={T['n_steps']}", f"train.n_train_itr={T['n_itr']}",
        f"train.save_model_freq={T['save_every']}", f"train.val_freq={T['val_freq']}",
    ]
    if kind == "robust":
        cmd.append("+env.wrappers.robomimic_lowdim.dr=true")
        cmd += [f"+env.wrappers.robomimic_lowdim.{k}={v}" for k, v in CFG["dr"].items()]
    return dict(name=f"train_{kind}_s{seed}", kind="train", cmd=cmd, logdir=logdir,
                done_file=os.path.join(logdir, "DONE.json"), deps=[], model=f"{kind}_s{seed}",
                final_ckpt=final_ckpt(kind, seed))


def model_ckpt(model):
    if model == "pretrained":
        return CFG["pretrained_ema_ckpt"], 0, []
    if model == "pretrained_raw":
        return CFG["pretrained_ckpt"], 0, []
    kind, seed = model.rsplit("_s", 1)
    return final_ckpt(kind, int(seed)), CFG["ft_denoising_steps"], [f"train_{model}"]


def eval_job(model, cond, video=False):
    ckpt, ftd, deps = model_ckpt(model)
    sub = "video" if video else "eval"
    logdir = os.path.join(P, "runs", sub, model, cond["name"])
    n_envs = E["video_envs"] if video else E["n_envs"]
    n_steps = EP_LEN * (1 if video else E["episodes_per_env"])
    cmd = [
        PY, "script/run.py",
        "--config-dir=cfg/robomimic/eval/square", "--config-name=eval_diffusion_mlp",
        f"base_policy_path={ckpt}", f"ft_denoising_steps={ftd}",
        "model._target_=model.diffusion.diffusion_eval.DiffusionEval",
        f"env.n_envs={n_envs}", f"n_steps={n_steps}", f"seed={E['seed']}", f"logdir={logdir}",
    ]
    cmd += [f"+env.wrappers.robomimic_lowdim.{k}={v}" for k, v in cond["overrides"].items()]
    if video:
        cmd += [f"render_num={n_envs}", "env.save_video=true"]
    return dict(name=f"{sub}_{model}__{cond['name']}", kind=sub, cmd=cmd, logdir=logdir,
                done_file=os.path.join(logdir, "result.npz"), deps=deps, model=model, cond=cond["name"])


def build_jobs():
    """Priority order. Role A = pretrained + DPPO side, role B = Robust DPPO side, ALL = both."""
    conds = CFG["conditions"]
    by_name = {c["name"]: c for c in conds}
    nominal, probe = by_name["nominal"], by_name.get("shift_4cm", conds[1])
    s0 = CFG["seeds_main"][0]
    # 1) sanity: pretrained policy (nominal + one perturbation) and the raw-weights check
    # 2) DPPO fine-tuning, then the full robustness sweep of pretrained and DPPO
    main_a = [eval_job("pretrained", nominal), eval_job("pretrained_raw", nominal), eval_job("pretrained", probe),
              train_job("dppo", s0)]
    main_a += [eval_job("pretrained", c) for c in conds if c["name"] not in ("nominal", probe["name"])]
    main_a += [eval_job(f"dppo_s{s0}", c) for c in conds]
    # 3) Robust DPPO (domain randomization during fine-tuning) and its sweep
    main_b = [train_job("robust", s0)] + [eval_job(f"robust_s{s0}", c) for c in conds]
    # 4) demo videos
    videos = [eval_job("pretrained", nominal, True), eval_job(f"dppo_s{s0}", nominal, True),
              eval_job(f"dppo_s{s0}", probe, True), eval_job(f"robust_s{s0}", probe, True)]
    # 5) extra training seeds (stretch goal)
    extra_a, extra_b = [], []
    for s in CFG["seeds_extra"]:
        extra_a += [train_job("dppo", s)] + [eval_job(f"dppo_s{s}", c) for c in conds]
        extra_b += [train_job("robust", s)] + [eval_job(f"robust_s{s}", c) for c in conds]
    if ROLE == "A":
        return main_a + videos + extra_a
    if ROLE == "B":
        return main_b + extra_b
    return main_a + main_b + videos + extra_a + extra_b


JOBS = build_jobs()
ALL_NAMES = {j["name"] for j in JOBS}


# State helpers
def is_done(job_or_name):
    if isinstance(job_or_name, str):  # a dependency, possibly owned by the other runtime
        if job_or_name.startswith("train_"):
            kind, seed = job_or_name[len("train_"):].rsplit("_s", 1)
            return os.path.exists(os.path.join(train_dir(kind, int(seed)), "DONE.json"))
        return False
    job = job_or_name
    if job["kind"] == "video":
        return os.path.exists(job["done_file"]) and bool(glob.glob(os.path.join(job["logdir"], "render", "*.mp4")))
    return os.path.exists(job["done_file"])


def skipped(job):
    return os.path.exists(os.path.join(PIPE_DIR, f"SKIP_{job['name']}"))


def tail(path, n_bytes=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def last_ckpt_itr(logdir):
    its = []
    for p in glob.glob(os.path.join(logdir, "checkpoint", "state_*.pt")):
        m = re.search(r"state_(\d+)\.pt$", p)
        if m:
            its.append(int(m.group(1)))
    return max(its) if its else None


def prune_checkpoints(logdir, keep=2):
    """Keep the newest `keep` checkpoints plus the final one, to save Drive space."""
    paths = []
    for p in glob.glob(os.path.join(logdir, "checkpoint", "state_*.pt")):
        m = re.search(r"state_(\d+)\.pt$", p)
        if m:
            paths.append((int(m.group(1)), p))
    paths.sort()
    for itr, p in paths[:-keep]:
        if itr != T["n_itr"] - 1:
            try:
                os.remove(p)
            except OSError:
                pass


def train_progress(job):
    log = tail(os.path.join(job["logdir"], "job.log"))
    train_itrs = [int(x) for x in re.findall(r"\] - (\d+): step\s+\d+", log)]
    evals = [float(x) for x in re.findall(r"eval: success rate\s+([0-9.]+)", log)]
    saved = last_ckpt_itr(job["logdir"])
    itr = max(train_itrs + ([saved] if saved is not None else []) + [-1])
    return itr, (evals[-1] if evals else None)


def eval_progress(job):
    log = tail(os.path.join(job["logdir"], "job.log"), 20_000)
    m = re.findall(r"Processed step (\d+) of (\d+)", log)
    return (int(m[-1][0]), int(m[-1][1])) if m else (0, None)


def eval_result(job):
    try:
        r = np.load(job["done_file"])
        n = int(r["num_episode"])
        k = int(round(float(r["eval_success_rate"]) * n))
        return f"{k}/{n} = {100.0 * k / max(n, 1):.1f}%"
    except Exception:
        return ""


# Status file
STATE = {j["name"]: dict(state="pending", tries=0, started=None, finished=None, seconds=None, note="") for j in JOBS}
if os.path.exists(STATUS_JSON):  # keep try counts and timings from earlier sessions
    try:
        old = json.load(open(STATUS_JSON)).get("jobs", {})
        for name, s in old.items():
            if name in STATE:
                # a job cut off by a disconnect (still "running") does not count as a failed try
                STATE[name]["tries"] = max(0, s.get("tries", 0) - (1 if s.get("state") == "running" else 0))
                STATE[name]["seconds"] = s.get("seconds")
                if s.get("state") == "failed":
                    STATE[name]["state"] = "failed"
                    STATE[name]["note"] = s.get("note", "")
    except Exception:
        pass
CURRENT = {}
HW = CFG.get("hardware", {})


def projected_finish():
    """Rough ETA for the remaining jobs, from timings measured in this study."""
    train_s = [s["seconds"] / T["n_itr"] for n, s in STATE.items() if n.startswith("train_") and s["state"] == "done" and s["seconds"]]
    eval_s = [s["seconds"] for n, s in STATE.items() if n.startswith("eval_") and s["state"] == "done" and s["seconds"]]
    per_itr = CURRENT.get("sec_per_itr") or (np.mean(train_s) if train_s else None)
    per_eval = np.mean(eval_s) if eval_s else None
    total = 0.0
    for j in JOBS:
        s = STATE[j["name"]]
        if s["state"] in ("done", "failed", "skipped", "blocked"):
            continue
        if j["name"] == CURRENT.get("name"):
            total += CURRENT.get("eta_s") or 0
        elif j["kind"] == "train":
            if per_itr is None:
                return None
            total += per_itr * T["n_itr"]
        else:
            if per_eval is None:
                return None
            total += per_eval
    return now() + dt.timedelta(seconds=total)


def write_status(final=False):
    rows = []
    for j in JOBS:
        s = STATE[j["name"]]
        if is_done(j) and s["state"] not in ("done",):
            s["state"] = "done"
        if skipped(j) and s["state"] == "pending":
            s["state"] = "skipped"
        info = s["note"]
        if s["state"] == "done" and j["kind"] in ("eval", "video"):
            info = eval_result(j)
        if j["name"] == CURRENT.get("name"):
            info = CURRENT.get("progress", "")
        dur = f"{s['seconds'] / 60:.1f} min" if s.get("seconds") else ""
        rows.append((j["name"], s["state"], s["tries"], dur, info))
    fin = projected_finish()
    counts = {k: sum(1 for r in rows if r[1] == k) for k in ("done", "running", "pending", "failed", "blocked", "skipped")}
    data = dict(role=ROLE, updated_at=stamp(), hardware=HW, current=CURRENT, counts=counts,
                projected_finish=stamp(fin) if fin else None, finished=final,
                jobs={j["name"]: STATE[j["name"]] for j in JOBS})
    tmp = STATUS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, STATUS_JSON)
    lines = [
        f"# Pipeline status (role {ROLE})",
        f"Updated: {stamp()} (Phoenix time) | {'FINISHED' if final else 'running'}",
        f"Hardware: {HW.get('gpu', '?')} | {HW.get('cpus', '?')} vCPUs | {HW.get('ram_gb', '?')} GB RAM",
        f"Jobs: {counts['done']} done, {counts['running']} running, {counts['pending']} pending, "
        f"{counts['failed']} failed, {counts['blocked']} blocked, {counts['skipped']} skipped",
        f"Projected finish of everything: {stamp(fin) if fin else 'not known yet (needs one finished job of each type)'}",
        "",
    ]
    if CURRENT:
        lines += [f"Now running: {CURRENT['name']} | {CURRENT.get('progress', '')} | started {CURRENT.get('started', '')}",
                  "Last log lines:", "```", *CURRENT.get("tail", []), "```", ""]
    lines += ["| job | state | tries | time | result / progress |", "|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    tmp = STATUS_MD + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, STATUS_MD)


# Running one job
def kill_group(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(CFG.get("kill_grace_seconds", 10))
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass


def run_job(job):
    s = STATE[job["name"]]
    os.makedirs(job["logdir"], exist_ok=True)
    logpath = os.path.join(job["logdir"], "job.log")
    if job["kind"] == "train":
        # keep in-training eval history from earlier segments before a resume overwrites result.pkl
        rp = os.path.join(job["logdir"], "result.pkl")
        if os.path.exists(rp):
            n = len(glob.glob(os.path.join(job["logdir"], "result_seg*.pkl")))
            shutil.move(rp, os.path.join(job["logdir"], f"result_seg{n}.pkl"))
    s["state"], s["tries"], s["started"] = "running", s["tries"] + 1, stamp()
    start_itr = (last_ckpt_itr(job["logdir"]) or -1) + 1 if job["kind"] == "train" else 0
    CURRENT.clear()
    CURRENT.update(name=job["name"], started=stamp(), progress="starting")
    event(f"START {job['name']} (try {s['tries']})" + (f" from iteration {start_itr}" if start_itr else ""))
    t0 = time.time()
    with open(logpath, "a") as lf:
        lf.write(f"\n===== {stamp()} start try {s['tries']} =====\n{' '.join(job['cmd'])}\n")
        lf.flush()
        p = subprocess.Popen(job["cmd"], cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
        last_size, last_change = -1, time.time()
        while p.poll() is None:
            time.sleep(CFG.get("poll_seconds", 30))
            size = os.path.getsize(logpath)
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > CFG["stall_minutes"] * 60:
                event(f"STALL {job['name']}: no log output for {CFG['stall_minutes']} min, killing it")
                kill_group(p)
                break
            if job["kind"] != "train" and time.time() - t0 > CFG["eval_timeout_minutes"] * 60:
                event(f"TIMEOUT {job['name']} after {CFG['eval_timeout_minutes']} min, killing it")
                kill_group(p)
                break
            el = time.time() - t0
            if job["kind"] == "train":
                itr, last_eval = train_progress(job)
                done_itrs = max(0, itr - start_itr + 1)
                spi = el / done_itrs if done_itrs > 0 else None
                eta = spi * (T["n_itr"] - itr - 1) if spi else None
                CURRENT.update(sec_per_itr=spi, eta_s=eta,
                               progress=f"iteration {max(itr, 0)}/{T['n_itr'] - 1}"
                               + (f", {spi / 60:.1f} min/iteration, ~{eta / 3600:.1f} h left" if spi else "")
                               + (f", last in-training eval success {100 * last_eval:.1f}%" if last_eval is not None else ""))
                prune_checkpoints(job["logdir"])
            else:
                step, total = eval_progress(job)
                CURRENT.update(eta_s=None, progress=f"policy step {step}/{total or '?'}")
            CURRENT["tail"] = [ln[:160] for ln in tail(logpath, 4000).strip().splitlines()[-6:]]
            write_status()
    rc = p.wait()
    secs = time.time() - t0
    ok = rc == 0 and (job["kind"] != "train" or os.path.exists(job["final_ckpt"]))
    if ok and job["kind"] == "train":
        prev = (s.get("seconds") or 0) if start_itr > 0 else 0
        s["seconds"] = prev + secs
        with open(job["done_file"], "w") as f:
            json.dump(dict(finished=stamp(), seconds_this_segment=secs, total_seconds=s["seconds"], hardware=HW, cmd=job["cmd"]), f, indent=1)
        prune_checkpoints(job["logdir"])
    elif ok:
        s["seconds"] = secs
    ok = ok and is_done(job)
    if ok:
        s["state"], s["finished"], s["note"] = "done", stamp(), ""
        event(f"DONE  {job['name']} in {secs / 60:.1f} min " + (eval_result(job) if job["kind"] != "train" else ""))
    else:
        err = [ln for ln in tail(logpath, 30000).splitlines() if re.search(r"Error|Exception|Traceback|Killed|CUDA", ln)]
        s["note"] = (err[-1][:200] if err else f"exit code {rc}")
        s["state"] = "failed" if s["tries"] >= CFG["max_tries"] else "pending"
        if job["kind"] == "train":  # keep the time spent so a resume can add to it
            s["seconds"] = (s.get("seconds") or 0) + secs
        event(f"FAIL  {job['name']} (try {s['tries']}/{CFG['max_tries']}, exit {rc}): {s['note']}")
    CURRENT.clear()
    write_status()
    return ok


# Main loop
def main():
    event(f"pipeline started, role {ROLE}, {len(JOBS)} jobs")
    for j in JOBS:  # a job that failed in an earlier session gets a fresh chance after a restart
        if STATE[j["name"]]["state"] == "failed" and STATE[j["name"]]["tries"] < CFG["max_tries"] + 1:
            STATE[j["name"]]["state"] = "pending"
    while True:
        if os.path.exists(os.path.join(PIPE_DIR, "STOP")):
            event("STOP file found, exiting")
            break
        write_status()
        todo = [j for j in JOBS if not is_done(j) and not skipped(j) and STATE[j["name"]]["state"] == "pending"]
        if not todo:
            break
        runnable, waiting = [], False
        for j in todo:
            deps_ok = all(is_done(d) for d in j["deps"])
            if deps_ok:
                runnable.append(j)
                continue
            dead = [d for d in j["deps"] if d in ALL_NAMES and STATE[d]["state"] in ("failed", "skipped")]
            if dead:
                STATE[j["name"]]["state"], STATE[j["name"]]["note"] = "blocked", f"needs {dead[0]}"
            else:
                waiting = True  # dependency still running here or on the other runtime
        if runnable:
            run_job(runnable[0])
        elif waiting:
            time.sleep(CFG.get("wait_seconds", 300))
        else:
            break
    write_status(final=True)
    event("pipeline finished")


if __name__ == "__main__":
    main()
