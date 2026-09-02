"""
Live GPT-4o ablation study for the K=6 Red-Team Attack Graph, rerun with the
same metric instrumentation as run_full_eval.py. Full horizon, no early break
(see run_full_eval.py docstring for why).

Note: because environment.py's transition kernel T is the identity matrix for
every action (patch/exploit transition dynamics are an unimplemented TODO),
AB2 (drop observation step) reduces here to a belief that never leaves the
uniform prior, and AB1 (drop prediction step) becomes nearly indistinguishable
from Standard_BSE since dropping an identity transition changes nothing. Both
are honest consequences of the environment's current simplification, not
metrics bugs -- the paper's prediction of "severe AB1 degradation on the
attack graph" does not hold in this simplified environment. Documented in the
README.

Run: python red_team_graph/run_ablations_eval.py
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
from red_team_graph.environment import create_k6_attack_graph
from eval_metrics import bootstrap_mean_ci, brier, nll, entropy, llm_call
from red_team_graph.run_full_eval import (SYSTEM_PROMPT, format_bse_prompt, state_to_index,
                                            R_SCAN, R_EXPLOIT_SUCCESS, R_EXPLOIT_FAIL, R_WAIT, R_EXPLOIT_REPEAT,
                                            FALSE_NEG_RATE, FALSE_POS_RATE, K, HORIZON, GAMMA, ACTIONS)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", date.today().isoformat())
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                     handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "attack_ablations_eval.log"), mode="w")])
logger = logging.getLogger("AttackAblationsEval")

MODEL_NAME = "gpt-4o"
N_ABLATION = int(os.environ.get("ATTACK_N_ABLATION", 25))
MAX_WORKERS = int(os.environ.get("ATTACK_MAX_WORKERS", 5))

MODEL = create_k6_attack_graph()
BELIEF_FILTER = BeliefFilter(MODEL.T, MODEL.Z)
PARSER = ExactMatchParser(ACTIONS)
CLIENT = OpenAI()

ABLATIONS = ["Standard_BSE", "AB1_No_Predict", "AB2_No_Observe", "AB9_High_Temp"]


def ablated_update(ablation, belief, act_idx, obs_idx):
    if ablation == "AB1_No_Predict":
        b_tilde = belief * MODEL.Z[act_idx, :, obs_idx]
        return b_tilde / np.sum(b_tilde)
    if ablation == "AB2_No_Observe":
        return np.dot(belief, MODEL.T[act_idx])
    return BELIEF_FILTER.update(belief, act_idx, obs_idx)


def _scan(rng, true_bits, target):
    is_vuln = true_bits[target] == 1
    p_flag = (1 - FALSE_NEG_RATE) if is_vuln else FALSE_POS_RATE
    return 1 if rng.random() < p_flag else 0


def run_ablation_episode(ablation, seed):
    rng = np.random.default_rng(seed)
    true_bits = tuple(int(x) for x in rng.integers(0, 2, size=K))
    true_idx = state_to_index(true_bits)
    belief = np.ones(len(MODEL.S)) / len(MODEL.S)
    action = "wait"
    obs_idx = 0
    temperature = 1.0 if ablation == "AB9_High_Temp" else 0.3
    ret_disc = ret_undisc = 0.0
    tokens = 0; latency = 0.0; calls = 0; abstentions = 0
    success_attempts = 0; failed_attempts = 0
    compromised = set(); first_intrusion_step = None
    brier_vals, nll_vals, ent_vals = [], [], []

    for t in range(HORIZON):
        act_idx = MODEL.A.index(action)
        belief = ablated_update(ablation, belief, act_idx, obs_idx)
        brier_vals.append(brier(belief, true_idx))
        nll_vals.append(nll(belief, true_idx))
        ent_vals.append(entropy(belief))

        prompt = format_bse_prompt(belief, compromised, steps_remaining=HORIZON - t)
        action, meta = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, temperature, PARSER, "wait")
        tokens += meta["prompt_tokens"] + meta["completion_tokens"]
        latency += meta["latency_s"]; calls += meta["calls"]; abstentions += int(meta["abstained"])

        if action.startswith("scan"):
            target = int(action.split("_")[1])
            obs_idx = _scan(rng, true_bits, target)
            ret_disc += R_SCAN * (GAMMA ** t); ret_undisc += R_SCAN
        elif action.startswith("exploit"):
            target = int(action.split("_")[1])
            if target in compromised:
                pass  # already owned: no reward, no re-count
            elif true_bits[target] == 1:
                success_attempts += 1
                compromised.add(target)
                if first_intrusion_step is None:
                    first_intrusion_step = t
                ret_disc += R_EXPLOIT_SUCCESS * (GAMMA ** t); ret_undisc += R_EXPLOIT_SUCCESS
            else:
                failed_attempts += 1
                ret_disc += R_EXPLOIT_FAIL * (GAMMA ** t); ret_undisc += R_EXPLOIT_FAIL
            obs_idx = 0
        else:
            ret_disc += R_WAIT * (GAMMA ** t); ret_undisc += R_WAIT
            obs_idx = 0

    return dict(ablation=ablation, ret_disc=ret_disc, ret_undisc=ret_undisc, tokens=tokens, latency=latency,
                calls=calls, abstentions=abstentions, success_attempts=success_attempts,
                failed_attempts=failed_attempts, compromised=list(compromised),
                first_intrusion_step=first_intrusion_step, brier=brier_vals, nll=nll_vals, entropy=ent_vals)


def aggregate(ablation, episodes):
    rets = [e["ret_disc"] for e in episodes]
    mean, lo, hi = bootstrap_mean_ci(rets)
    total_success = sum(e["success_attempts"] for e in episodes)
    total_failed = sum(e["failed_attempts"] for e in episodes)
    intrusion_steps = [e["first_intrusion_step"] for e in episodes if e["first_intrusion_step"] is not None]
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
        "avg_unique_nodes_compromised": float(np.mean([len(e["compromised"]) for e in episodes])),
        "exploitation_success_rate": (total_success / (total_success + total_failed)
                                       if (total_success + total_failed) > 0 else None),
        "avg_steps_to_first_intrusion": float(np.mean(intrusion_steps)) if intrusion_steps else None,
        "avg_network_compromise_coverage": float(np.mean([len(e["compromised"]) / K for e in episodes])),
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

    out_path = os.path.join(LOG_DIR, "attack_ablations_eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"config": {"n_ablation": N_ABLATION, "horizon": HORIZON, "gamma": GAMMA, "model": MODEL_NAME},
                    "results": results}, f, indent=2)
    logger.info("Saved results to %s", out_path)
    for ablation, r in results.items():
        logger.info("%s: %s", ablation, json.dumps(r))
