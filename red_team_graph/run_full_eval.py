"""
Live GPT-4o evaluation for the K=6 Red-Team Attack Graph producing the numbers
reported in README.md Section 10, using the metric definitions in the
accompanying paper (Section VI.C).

Two fixes relative to the original red_team_graph/run_attack.py:
  1. Episodes no longer break after the first exploit attempt. The original
     script terminated the episode on any exploit (success or fail), which
     makes "network compromise coverage" and "avg nodes compromised" (plural)
     impossible to measure honestly. Episodes here run the full horizon.
  2. The reactive baseline's prompt now includes which node was scanned and
     the raw scan verdict, since a bare "observation: 0/1" with no node id is
     uninterpretable.

Known limitation carried over from environment.py: the `patch` action and the
node "dependencies" (lateral-movement topology) are unimplemented placeholders
in the environment (see the `# TODO` there). We exclude `patch` from the
action menu offered to the LLM rather than silently simulate a fake effect.

Run: python red_team_graph/run_full_eval.py
"""
import os
import re
import sys
import json
import logging
import itertools
from datetime import date
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_engine import BeliefFilter, ExactMatchParser
from red_team_graph.environment import create_k6_attack_graph
from eval_metrics import bootstrap_mean_ci, brier, nll, entropy, jsd, action_distribution, llm_call

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", date.today().isoformat())
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                     handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "attack_full_eval.log"), mode="w")])
logger = logging.getLogger("AttackFullEval")

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.3
K = 6
HORIZON = 30            # matches paper Sec. VI.A.b spec (T=30)
GAMMA = 0.95
N_MAIN = int(os.environ.get("ATTACK_N_MAIN", 40))
MAX_WORKERS = int(os.environ.get("ATTACK_MAX_WORKERS", 5))
N_JSD_GROUPS = int(os.environ.get("ATTACK_N_JSD_GROUPS", 8))
K_JSD_SAMPLES = int(os.environ.get("ATTACK_K_JSD_SAMPLES", 5))

# Reward constants per paper Sec. VI.A.b (differ from the original demo script,
# which used an undocumented R(wait)=-1 and had no patch reward).
R_SCAN = -0.5
R_EXPLOIT_SUCCESS = 20.0
R_EXPLOIT_FAIL = -5.0
R_WAIT = -0.1
R_EXPLOIT_REPEAT = 0.0   # re-exploiting an already-compromised node pays nothing (no reward-farming)
FALSE_NEG_RATE = 0.15   # Z(1|vulnerable) = 1 - 0.15 = 0.85
FALSE_POS_RATE = 0.10   # Z(1|hardened)  = 0.10

MODEL = create_k6_attack_graph()
BELIEF_FILTER = BeliefFilter(MODEL.T, MODEL.Z)
# `patch` is excluded: its transition dynamics are an unimplemented TODO in environment.py.
ACTIONS = [a for a in MODEL.A if not a.startswith("patch")]
PARSER = ExactMatchParser(ACTIONS)
CLIENT = OpenAI()

SYSTEM_PROMPT = (
    "Select the action that maximizes your expected discounted return over the remaining "
    "steps of the episode, not just the immediate reward -- scanning now can be worth a "
    "short-term cost if it avoids a larger expected loss later. Use the information provided "
    "below. Break ties uniformly. Output your reasoning, then a final line strictly formatted "
    "as 'ACTION: <action>'."
)


def state_to_index(state_bits):
    idx = 0
    for b in state_bits:
        idx = (idx << 1) | int(b)
    return idx


def node_marginals(belief):
    probs = np.zeros(K)
    for node in range(K):
        probs[node] = sum(belief[s_idx] for s_idx, state in enumerate(MODEL.S) if state[node] == 1)
    return probs


def _compromised_line(compromised):
    if not compromised:
        return "No nodes compromised yet."
    return (f"Already compromised (further exploits on these yield {R_EXPLOIT_REPEAT:.0f} reward): "
            f"{', '.join(str(n) for n in sorted(compromised))}.")


def format_bse_prompt(belief, compromised=(), steps_remaining=None):
    probs = node_marginals(belief)
    lines = [
        "DOMAIN: K=6 Red-Team Attack Graph\n",
        "GOAL: Discover vulnerable entry points and execute successful exploits.\n",
        f"REWARDS: +{R_EXPLOIT_SUCCESS:.0f} for successful exploit, {R_EXPLOIT_FAIL:.0f} for failed exploit, "
        f"{R_SCAN:.1f} for scan, {R_WAIT:.1f} for wait.\n",
        f"STEPS REMAINING: {steps_remaining}\n",
        f"AVAILABLE ACTIONS: {', '.join(ACTIONS)}\n\n",
        f"{_compromised_line(compromised)}\n\n",
        "CURRENT MARGINAL PROBABILITY OF EACH NODE BEING VULNERABLE:\n",
    ]
    for i, p in enumerate(probs):
        lines.append(f"- Node {i}: {p:.4f}\n")
    return "".join(lines)


def format_reactive_prompt(last_scan_target, last_scan_result, compromised=(), steps_remaining=None):
    if last_scan_target is None:
        scan_line = "No scans performed yet."
    else:
        scan_line = f"Scanned node {last_scan_target}, result={last_scan_result} (1=flagged vulnerable, 0=clean)."
    return (
        "DOMAIN: K=6 Red-Team Attack Graph\n"
        "GOAL: Discover vulnerable entry points and execute successful exploits.\n"
        f"REWARDS: +{R_EXPLOIT_SUCCESS:.0f} for successful exploit, {R_EXPLOIT_FAIL:.0f} for failed exploit, "
        f"{R_SCAN:.1f} for scan, {R_WAIT:.1f} for wait.\n"
        f"STEPS REMAINING: {steps_remaining}\n"
        f"AVAILABLE ACTIONS: {', '.join(ACTIONS)}\n\n"
        f"{_compromised_line(compromised)}\n\n"
        f"LATEST SCAN OBSERVATION: {scan_line}\n"
    )


def format_nl_prompt(prev_belief_text, last_scan_target, last_scan_result, compromised=(), steps_remaining=None):
    if last_scan_target is None:
        scan_line = "No scans performed yet."
    else:
        scan_line = f"Scanned node {last_scan_target}, result={last_scan_result} (1=flagged vulnerable, 0=clean)."
    return (
        "DOMAIN: K=6 Red-Team Attack Graph\n"
        "GOAL: Discover vulnerable entry points and execute successful exploits.\n"
        f"REWARDS: +{R_EXPLOIT_SUCCESS:.0f} for successful exploit, {R_EXPLOIT_FAIL:.0f} for failed exploit, "
        f"{R_SCAN:.1f} for scan, {R_WAIT:.1f} for wait.\n"
        f"STEPS REMAINING: {steps_remaining}\n"
        f"AVAILABLE ACTIONS: {', '.join(ACTIONS)}\n\n"
        f"{_compromised_line(compromised)}\n\n"
        f"YOUR PREVIOUS BELIEF SUMMARY:\n{prev_belief_text}\n\n"
        f"LATEST SCAN OBSERVATION: {scan_line}\n\n"
        "INSTRUCTION: First write one line starting with 'BELIEF:' that states, for each of the "
        "6 nodes, your estimated probability it is vulnerable (e.g. 'BELIEF: Node 0: 70%, Node 1: 20%, ...'). "
        "Then output a final line strictly formatted as 'ACTION: <action>'."
    )


def parse_belief_line(text):
    m = re.search(r"BELIEF\s*[:\-]?\s*(.+?)(?:\n\s*ACTION\s*:|\Z)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text


def parse_nl_node_probs(text):
    probs = np.full(K, 0.5)
    found = 0
    for node in range(K):
        m = re.search(rf"node\s*{node}\b[^0-9%]{{0,15}}?(\d{{1,3}}(?:\.\d+)?)\s*%", text, re.IGNORECASE)
        if m:
            probs[node] = float(m.group(1)) / 100.0
            found += 1
            continue
        m = re.search(rf"node\s*{node}\b[^0-9]{{0,10}}?(0?\.\d+|1\.0)", text, re.IGNORECASE)
        if m:
            probs[node] = float(m.group(1))
            found += 1
    return probs, found >= K // 2


def _scan(rng, true_bits, target):
    is_vuln = true_bits[target] == 1
    p_flag = (1 - FALSE_NEG_RATE) if is_vuln else FALSE_POS_RATE
    return 1 if rng.random() < p_flag else 0


def _empty_episode_state():
    return dict(ret_disc=0.0, ret_undisc=0.0, tokens=0, latency=0.0, calls=0, abstentions=0,
                success_attempts=0, failed_attempts=0, compromised=set(), first_intrusion_step=None,
                brier=[], nll=[], entropy=[], consistency=[])


def run_episode(method, seed):
    rng = np.random.default_rng(seed)
    true_bits = tuple(int(x) for x in rng.integers(0, 2, size=K))
    true_idx = state_to_index(true_bits)
    belief = np.ones(len(MODEL.S)) / len(MODEL.S)
    action = "wait"
    obs_idx = 0
    last_scan_target, last_scan_result = None, None
    belief_text = "BELIEF: no scans yet, all nodes 50% likely vulnerable."
    malformed = 0
    st = _empty_episode_state()

    for t in range(HORIZON):
        act_idx = MODEL.A.index(action)
        belief = BELIEF_FILTER.update(belief, act_idx, obs_idx)
        st["brier"].append(brier(belief, true_idx))
        st["nll"].append(nll(belief, true_idx))
        st["entropy"].append(entropy(belief))

        if method == "Reactive":
            prompt = format_reactive_prompt(last_scan_target, last_scan_result, st["compromised"],
                                             steps_remaining=HORIZON - t)
            action, meta = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, TEMPERATURE, PARSER, "wait")
        elif method == "BSE":
            prompt = format_bse_prompt(belief, st["compromised"], steps_remaining=HORIZON - t)
            action, meta = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, TEMPERATURE, PARSER, "wait")
        else:  # NL-Tracker
            prompt = format_nl_prompt(belief_text, last_scan_target, last_scan_result, st["compromised"],
                                       steps_remaining=HORIZON - t)
            action, meta = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, TEMPERATURE, PARSER, "wait",
                                     return_text=True)
            belief_text = parse_belief_line(meta["text"]) if meta["text"] else belief_text
            _, ok = parse_nl_node_probs(belief_text)
            if not ok:
                malformed += 1

        st["tokens"] += meta["prompt_tokens"] + meta["completion_tokens"]
        st["latency"] += meta["latency_s"]; st["calls"] += meta["calls"]; st["abstentions"] += int(meta["abstained"])
        st["consistency"].append((tuple(np.round(node_marginals(belief), 2)), prompt))

        if action.startswith("scan"):
            target = int(action.split("_")[1])
            obs_idx = _scan(rng, true_bits, target)
            last_scan_target, last_scan_result = target, obs_idx
            st["ret_disc"] += R_SCAN * (GAMMA ** t); st["ret_undisc"] += R_SCAN
        elif action.startswith("exploit"):
            target = int(action.split("_")[1])
            if target in st["compromised"]:
                pass  # already owned: no reward, no re-count (prevents reward-farming a known node)
            elif true_bits[target] == 1:
                st["success_attempts"] += 1
                st["compromised"].add(target)
                if st["first_intrusion_step"] is None:
                    st["first_intrusion_step"] = t
                st["ret_disc"] += R_EXPLOIT_SUCCESS * (GAMMA ** t); st["ret_undisc"] += R_EXPLOIT_SUCCESS
            else:
                st["failed_attempts"] += 1
                st["ret_disc"] += R_EXPLOIT_FAIL * (GAMMA ** t); st["ret_undisc"] += R_EXPLOIT_FAIL
            obs_idx = 0  # exploit yields no scan information; keep belief update a no-op next step
        else:  # wait
            st["ret_disc"] += R_WAIT * (GAMMA ** t); st["ret_undisc"] += R_WAIT
            obs_idx = 0

    st["method"] = method
    st["malformed"] = malformed
    st["compromised"] = list(st["compromised"])
    return st


def aggregate(method_name, episodes):
    rets = [e["ret_disc"] for e in episodes]
    mean, lo, hi = bootstrap_mean_ci(rets)
    n = len(episodes)
    total_success = sum(e["success_attempts"] for e in episodes)
    total_failed = sum(e["failed_attempts"] for e in episodes)
    intrusion_steps = [e["first_intrusion_step"] for e in episodes if e["first_intrusion_step"] is not None]
    all_brier = [b for e in episodes for b in e["brier"]]
    all_nll = [v for e in episodes for v in e["nll"]]
    all_ent = [v for e in episodes for v in e["entropy"]]
    return {
        "method": method_name,
        "n_episodes": n,
        "mean_discounted_return": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "std_discounted_return": float(np.std(rets)),
        "avg_unique_nodes_compromised": float(np.mean([len(e["compromised"]) for e in episodes])),
        "avg_successful_exploit_attempts": float(np.mean([e["success_attempts"] for e in episodes])),
        "avg_failed_exploit_attempts": float(np.mean([e["failed_attempts"] for e in episodes])),
        "exploitation_success_rate": (total_success / (total_success + total_failed)
                                       if (total_success + total_failed) > 0 else None),
        "episodes_with_any_intrusion": len(intrusion_steps),
        "avg_steps_to_first_intrusion": float(np.mean(intrusion_steps)) if intrusion_steps else None,
        "avg_network_compromise_coverage": float(np.mean([len(e["compromised"]) / K for e in episodes])),
        "mean_brier": float(np.mean(all_brier)),
        "mean_nll": float(np.mean(all_nll)),
        "mean_entropy": float(np.mean(all_ent)),
        "avg_tokens_per_episode": float(np.mean([e["tokens"] for e in episodes])),
        "avg_latency_s_per_episode": float(np.mean([e["latency"] for e in episodes])),
        "avg_calls_per_episode": float(np.mean([e["calls"] for e in episodes])),
        "abstention_rate": float(sum(e["abstentions"] for e in episodes) / sum(e["calls"] for e in episodes)),
        "malformed_belief_rate": (float(sum(e.get("malformed", 0) for e in episodes) /
                                         sum(e["calls"] for e in episodes)) if "malformed" in episodes[0] else None),
    }


def measure_decision_consistency(method_name, episodes):
    groups = defaultdict(set)
    for e in episodes:
        for belief_key, prompt in e["consistency"]:
            groups[belief_key].add(prompt)
    candidate_groups = [g for g in groups.values() if len(g) >= 2]
    candidate_groups.sort(key=lambda g: -len(g))
    candidate_groups = candidate_groups[:N_JSD_GROUPS]

    jsd_values = []
    for group in candidate_groups:
        prompts = list(group)[:3]
        dists = []
        for prompt in prompts:
            samples = []
            for _ in range(K_JSD_SAMPLES):
                action, _ = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, TEMPERATURE, PARSER, "wait")
                samples.append(action)
            dists.append(action_distribution(samples, ACTIONS))
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                jsd_values.append(jsd(dists[i], dists[j]))

    if not jsd_values:
        return {"method": method_name, "n_pairs": 0, "median_jsd": None, "iqr_jsd": None, "max_jsd": None}
    arr = np.array(jsd_values)
    return {
        "method": method_name,
        "n_pairs": len(arr),
        "median_jsd": float(np.median(arr)),
        "iqr_jsd": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "max_jsd": float(np.max(arr)),
    }


def run_method(method_name, n=N_MAIN):
    logger.info("Running %s (%d seeds, horizon=%d)...", method_name, n, HORIZON)
    episodes = [None] * n
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_episode, method_name, seed): seed for seed in range(n)}
        done = 0
        for fut in as_completed(futures):
            seed = futures[fut]
            episodes[seed] = fut.result()
            done += 1
            if done % 5 == 0 or done == n:
                logger.info("%s: completed %d/%d episodes", method_name, done, n)
    return episodes


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        logger.critical("FATAL: OPENAI_API_KEY missing.")
        sys.exit(1)

    results = {}
    episode_cache = {}
    for name in ["Reactive", "BSE", "NL-Tracker"]:
        eps = run_method(name, N_MAIN)
        episode_cache[name] = eps
        results[name] = aggregate(name, eps)

    logger.info("Measuring decision consistency (JSD) for each method...")
    consistency = {}
    for name in ["Reactive", "BSE", "NL-Tracker"]:
        consistency[name] = measure_decision_consistency(name, episode_cache[name])

    out_path = os.path.join(LOG_DIR, "attack_full_eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"config": {"n_main": N_MAIN, "horizon": HORIZON, "gamma": GAMMA, "temperature": TEMPERATURE,
                               "model": MODEL_NAME, "k_jsd_samples": K_JSD_SAMPLES, "n_jsd_groups": N_JSD_GROUPS,
                               "actions_excluded": "patch_* (unimplemented transition dynamics)"},
                    "results": results, "consistency": consistency}, f, indent=2)
    logger.info("Saved results to %s", out_path)
    for name, r in results.items():
        logger.info("%s: %s", name, json.dumps(r))
    for name, c in consistency.items():
        logger.info("%s consistency: %s", name, json.dumps(c))
