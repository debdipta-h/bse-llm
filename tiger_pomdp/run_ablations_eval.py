"""
Live GPT-4o ablation study for the Tiger POMDP, rerun with the same metric
instrumentation as run_full_eval.py (task return w/ bootstrap CI, belief
calibration, compute cost). Ablations AB1 (drop prediction step) and AB2
(drop observation step) show the effect of a *wrong* belief actually shown to
the LLM, so calibration here is computed against the ablated belief itself,
not the true canonical posterior.

Run: python tiger_pomdp/run_ablations_eval.py
"""
import os
import sys
import json
import logging
from datetime import date
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_engine import BeliefFilter, ExactMatchParser
from tiger_pomdp.environment import create_tiger_env
from eval_metrics import bootstrap_mean_ci, brier, nll, entropy, llm_call
from tiger_pomdp.run_full_eval import SYSTEM_PROMPT, format_bse_prompt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", date.today().isoformat())
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                     handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "tiger_ablations_eval.log"), mode="w")])
logger = logging.getLogger("TigerAblationsEval")

MODEL_NAME = "gpt-4o"
HORIZON = 20
GAMMA = 0.95
N_ABLATION = int(os.environ.get("TIGER_N_ABLATION", 25))
MAX_WORKERS = int(os.environ.get("TIGER_MAX_WORKERS", 5))

MODEL = create_tiger_env()
BELIEF_FILTER = BeliefFilter(MODEL.T, MODEL.Z)
PARSER = ExactMatchParser(MODEL.A)
CLIENT = OpenAI()

ABLATIONS = ["Standard_BSE", "AB1_No_Predict", "AB2_No_Observe", "AB9_High_Temp"]


def ablated_update(ablation, belief, act_idx, obs_idx):
    if ablation == "AB1_No_Predict":
        # Correction only: skip the transition push-through.
        b_tilde = belief * MODEL.Z[act_idx, :, obs_idx]
        return b_tilde / np.sum(b_tilde)
    if ablation == "AB2_No_Observe":
        # Prediction only: skip the observation-conditioned correction.
        return np.dot(belief, MODEL.T[act_idx])
    return BELIEF_FILTER.update(belief, act_idx, obs_idx)


def run_ablation_episode(ablation, seed):
    rng = np.random.default_rng(seed)
    true_state = int(rng.integers(2))
    obs_idx = true_state if rng.random() < 0.85 else 1 - true_state
    belief = np.array([0.5, 0.5])
    action = "listen"
    temperature = 1.0 if ablation == "AB9_High_Temp" else 0.3
    ret_disc = ret_undisc = 0.0
    listens = 0
    tokens = 0
    latency = 0.0
    calls = 0
    abstentions = 0
    brier_vals, nll_vals, ent_vals = [], [], []

    for t in range(HORIZON):
        act_idx = MODEL.A.index(action)
        belief = ablated_update(ablation, belief, act_idx, obs_idx)
        brier_vals.append(brier(belief, true_state))
        nll_vals.append(nll(belief, true_state))
        ent_vals.append(entropy(belief))

        prompt = format_bse_prompt(belief)
        action, meta = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, temperature, PARSER, "listen")
        tokens += meta["prompt_tokens"] + meta["completion_tokens"]
        latency += meta["latency_s"]; calls += meta["calls"]; abstentions += int(meta["abstained"])

        if action == "listen":
            ret_disc += -1.0 * (GAMMA ** t); ret_undisc += -1.0; listens += 1
            obs_idx = true_state if rng.random() < 0.85 else 1 - true_state
        else:
            won = (action == "open-right" and true_state == 0) or (action == "open-left" and true_state == 1)
            r = 10.0 if won else -100.0
            ret_disc += r * (GAMMA ** t); ret_undisc += r
            return dict(ablation=ablation, outcome="treasure" if won else "eaten", ret_disc=ret_disc,
                        ret_undisc=ret_undisc, listens=listens, steps=t + 1, tokens=tokens, latency=latency,
                        calls=calls, abstentions=abstentions, brier=brier_vals, nll=nll_vals, entropy=ent_vals)

    return dict(ablation=ablation, outcome="timeout", ret_disc=ret_disc, ret_undisc=ret_undisc, listens=listens,
                steps=HORIZON, tokens=tokens, latency=latency, calls=calls, abstentions=abstentions,
                brier=brier_vals, nll=nll_vals, entropy=ent_vals)


def aggregate(ablation, episodes):
    rets = [e["ret_disc"] for e in episodes]
    mean, lo, hi = bootstrap_mean_ci(rets)
    outcomes = defaultdict(int)
    for e in episodes:
        outcomes[e["outcome"]] += 1
    all_brier = [b for e in episodes for b in e["brier"]]
    all_nll = [v for e in episodes for v in e["nll"]]
    all_ent = [v for e in episodes for v in e["entropy"]]
    return {
        "ablation": ablation,
        "n_episodes": len(episodes),
        "mean_discounted_return": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "std_discounted_return": float(np.std(rets)),
        "outcomes": dict(outcomes),
        "avg_listens": float(np.mean([e["listens"] for e in episodes])),
        "mean_brier": float(np.mean(all_brier)),
        "mean_nll": float(np.mean(all_nll)),
        "mean_entropy": float(np.mean(all_ent)),
        "avg_tokens_per_episode": float(np.mean([e["tokens"] for e in episodes])),
        "avg_calls_per_episode": float(np.mean([e["calls"] for e in episodes])),
        "abstention_rate": float(sum(e["abstentions"] for e in episodes) / sum(e["calls"] for e in episodes)),
    }


def run_ablation(ablation, n=N_ABLATION):
    logger.info("Running %s (%d seeds)...", ablation, n)
    episodes = [None] * n
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_ablation_episode, ablation, seed): seed for seed in range(n)}
        done = 0
        for fut in as_completed(futures):
            seed = futures[fut]
            episodes[seed] = fut.result()
            done += 1
            if done % 5 == 0 or done == n:
                logger.info("%s: completed %d/%d episodes", ablation, done, n)
    return episodes


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        logger.critical("FATAL: OPENAI_API_KEY missing.")
        sys.exit(1)

    results = {}
    for ablation in ABLATIONS:
        eps = run_ablation(ablation)
        results[ablation] = aggregate(ablation, eps)

    out_path = os.path.join(LOG_DIR, "tiger_ablations_eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"config": {"n_ablation": N_ABLATION, "horizon": HORIZON, "gamma": GAMMA, "model": MODEL_NAME},
                    "results": results}, f, indent=2)
    logger.info("Saved results to %s", out_path)
    for ablation, r in results.items():
        logger.info("%s: %s", ablation, json.dumps(r))
