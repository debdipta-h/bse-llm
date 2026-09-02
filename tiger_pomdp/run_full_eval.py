"""
Live GPT-4o evaluation for the Tiger POMDP producing the numbers reported in
README.md Section 10, following the metric definitions in the accompanying
paper (Section VI.C): task return (with bootstrap CI), belief calibration
(Brier / NLL / entropy), decision consistency (Jensen-Shannon divergence over
belief-equivalent history pairs), and compute cost (tokens / latency /
abstentions).

Three methods are evaluated (see README for why CoT / ReAct / QMDP / POMCP
are out of scope for this run):
  - Reactive: prompt conditioned on the latest observation only.
  - BSE: prompt conditioned on the exact Bayes-filter posterior.
  - NL-Tracker: prompt conditioned on a free-text belief the LLM itself
    maintains and rewrites every step (isolates "any belief" vs "probabilistic
    belief").

Run: python tiger_pomdp/run_full_eval.py
"""
import os
import re
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
from eval_metrics import bootstrap_mean_ci, brier, nll, entropy, jsd, action_distribution, llm_call

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", date.today().isoformat())
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                     handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "tiger_full_eval.log"), mode="w")])
logger = logging.getLogger("TigerFullEval")

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.3
HORIZON = 20            # matches README Section 4 spec (T=20)
GAMMA = 0.95
N_MAIN = int(os.environ.get("TIGER_N_MAIN", 40))
MAX_WORKERS = int(os.environ.get("TIGER_MAX_WORKERS", 5))
N_JSD_GROUPS = int(os.environ.get("TIGER_N_JSD_GROUPS", 10))
K_JSD_SAMPLES = int(os.environ.get("TIGER_K_JSD_SAMPLES", 5))

MODEL = create_tiger_env()
BELIEF_FILTER = BeliefFilter(MODEL.T, MODEL.Z)
PARSER = ExactMatchParser(MODEL.A)
CLIENT = OpenAI()

# Identical across all three methods (paper Sec. VI.B: baselines share system prompt,
# temperature, and parser -- only the state representation in the user prompt varies).
SYSTEM_PROMPT = (
    "Select the action that maximizes the expected immediate reward given the "
    "information provided below. Break ties uniformly. Output your reasoning, then "
    "a final line strictly formatted as 'ACTION: <action>'."
)


def format_bse_prompt(belief):
    return (
        "DOMAIN: Tiger POMDP\n"
        "GOAL: Choose the door with treasure while avoiding the tiger.\n"
        "REWARDS: +10 for treasure, -100 for tiger, -1 for listening.\n"
        f"AVAILABLE ACTIONS: {', '.join(MODEL.A)}\n\n"
        "CURRENT BELIEF POSTERIOR:\n"
        f"- tiger-left:  {belief[0]:.4f}\n"
        f"- tiger-right: {belief[1]:.4f}\n"
    )


def format_reactive_prompt(obs_name):
    return (
        "DOMAIN: Tiger POMDP\n"
        "GOAL: Choose the door without the tiger.\n"
        "REWARDS: +10 for treasure, -100 for tiger, -1 for listening.\n"
        f"AVAILABLE ACTIONS: {', '.join(MODEL.A)}\n\n"
        f"LATEST SENSOR OBSERVATION: {obs_name}\n"
    )


def format_nl_prompt(prev_belief_text, last_action, last_obs):
    return (
        "DOMAIN: Tiger POMDP\n"
        "GOAL: Choose the door with treasure while avoiding the tiger.\n"
        "REWARDS: +10 for treasure, -100 for tiger, -1 for listening.\n"
        f"AVAILABLE ACTIONS: {', '.join(MODEL.A)}\n\n"
        f"YOUR PREVIOUS BELIEF SUMMARY:\n{prev_belief_text}\n\n"
        f"LATEST ACTION-OBSERVATION: action={last_action}, observation={last_obs}\n\n"
        "INSTRUCTION: First write one line starting with 'BELIEF:' that states your updated "
        "belief in natural language, including your best estimate of the probability the "
        "tiger is on the left vs. the right (e.g., 'BELIEF: ~80% left, ~20% right, based on ...'). "
        "Then output a final line strictly formatted as 'ACTION: <action>'."
    )


def _find_prob(text, keyword):
    # keyword before number: "left ... 80%" or "left: 0.8"
    m = re.search(rf"{keyword}[^0-9%]{{0,25}}?(\d{{1,3}}(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(rf"{keyword}[^0-9]{{0,20}}?(0?\.\d+|1\.0)\b", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # number before keyword: "80% left"
    m = re.search(rf"(\d{{1,3}}(?:\.\d+)?)\s*%[^a-zA-Z]{{0,12}}{keyword}", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0
    return None


def parse_nl_belief(text):
    pl = _find_prob(text, "left")
    pr = _find_prob(text, "right")
    if pl is not None and pr is not None:
        s = pl + pr
        if s > 0:
            return np.array([pl / s, pr / s]), True
    return np.array([0.5, 0.5]), False


def parse_belief_line(text):
    # Tolerate markdown (**BELIEF:**), leading punctuation, and multi-line belief text
    # that runs up to the next ACTION: line.
    m = re.search(r"BELIEF\s*[:\-]?\s*(.+?)(?:\n\s*ACTION\s*:|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Episode runners. Each returns a per-episode metrics dict. A parallel "genie"
# Bayes belief is tracked for every method (evaluator-only, never shown to
# Reactive/NL-Tracker) so calibration and decision-consistency can be scored
# against the true canonical posterior, per the paper's protocol.
# ---------------------------------------------------------------------------

def _step_env(rng, true_state, obs_idx):
    return true_state if rng.random() < 0.85 else 1 - true_state


def run_reactive_episode(seed):
    rng = np.random.default_rng(seed)
    true_state = int(rng.integers(2))
    obs_idx = _step_env(rng, true_state, None)
    genie_belief = np.array([0.5, 0.5])
    action = "listen"
    ret_disc = ret_undisc = 0.0
    listens = 0
    tokens = 0
    latency = 0.0
    calls = 0
    abstentions = 0
    brier_vals, nll_vals, ent_vals = [], [], []
    consistency_records = []

    for t in range(HORIZON):
        act_idx = MODEL.A.index(action)
        genie_belief = BELIEF_FILTER.update(genie_belief, act_idx, obs_idx)
        brier_vals.append(brier(genie_belief, true_state))
        nll_vals.append(nll(genie_belief, true_state))
        ent_vals.append(entropy(genie_belief))

        obs_name = "hear-left" if obs_idx == 0 else "hear-right"
        prompt = format_reactive_prompt(obs_name)
        action, meta = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, TEMPERATURE, PARSER, "listen")
        tokens += meta["prompt_tokens"] + meta["completion_tokens"]
        latency += meta["latency_s"]; calls += meta["calls"]; abstentions += int(meta["abstained"])
        consistency_records.append((tuple(np.round(genie_belief, 2)), prompt))

        if action == "listen":
            ret_disc += -1.0 * (GAMMA ** t); ret_undisc += -1.0; listens += 1
            obs_idx = _step_env(rng, true_state, obs_idx)
        else:
            won = (action == "open-right" and true_state == 0) or (action == "open-left" and true_state == 1)
            r = 10.0 if won else -100.0
            ret_disc += r * (GAMMA ** t); ret_undisc += r
            return dict(method="Reactive", outcome="treasure" if won else "eaten", ret_disc=ret_disc,
                        ret_undisc=ret_undisc, listens=listens, steps=t + 1, tokens=tokens, latency=latency,
                        calls=calls, abstentions=abstentions, brier=brier_vals, nll=nll_vals, entropy=ent_vals,
                        consistency=consistency_records)

    return dict(method="Reactive", outcome="timeout", ret_disc=ret_disc, ret_undisc=ret_undisc, listens=listens,
                steps=HORIZON, tokens=tokens, latency=latency, calls=calls, abstentions=abstentions,
                brier=brier_vals, nll=nll_vals, entropy=ent_vals, consistency=consistency_records)


def run_bse_episode(seed):
    rng = np.random.default_rng(seed)
    true_state = int(rng.integers(2))
    obs_idx = _step_env(rng, true_state, None)
    belief = np.array([0.5, 0.5])
    action = "listen"
    ret_disc = ret_undisc = 0.0
    listens = 0
    tokens = 0
    latency = 0.0
    calls = 0
    abstentions = 0
    brier_vals, nll_vals, ent_vals = [], [], []
    consistency_records = []

    for t in range(HORIZON):
        act_idx = MODEL.A.index(action)
        belief = BELIEF_FILTER.update(belief, act_idx, obs_idx)
        brier_vals.append(brier(belief, true_state))
        nll_vals.append(nll(belief, true_state))
        ent_vals.append(entropy(belief))

        prompt = format_bse_prompt(belief)
        action, meta = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, TEMPERATURE, PARSER, "listen")
        tokens += meta["prompt_tokens"] + meta["completion_tokens"]
        latency += meta["latency_s"]; calls += meta["calls"]; abstentions += int(meta["abstained"])
        consistency_records.append((tuple(np.round(belief, 2)), prompt))

        if action == "listen":
            ret_disc += -1.0 * (GAMMA ** t); ret_undisc += -1.0; listens += 1
            obs_idx = _step_env(rng, true_state, obs_idx)
        else:
            won = (action == "open-right" and true_state == 0) or (action == "open-left" and true_state == 1)
            r = 10.0 if won else -100.0
            ret_disc += r * (GAMMA ** t); ret_undisc += r
            return dict(method="BSE", outcome="treasure" if won else "eaten", ret_disc=ret_disc,
                        ret_undisc=ret_undisc, listens=listens, steps=t + 1, tokens=tokens, latency=latency,
                        calls=calls, abstentions=abstentions, brier=brier_vals, nll=nll_vals, entropy=ent_vals,
                        consistency=consistency_records)

    return dict(method="BSE", outcome="timeout", ret_disc=ret_disc, ret_undisc=ret_undisc, listens=listens,
                steps=HORIZON, tokens=tokens, latency=latency, calls=calls, abstentions=abstentions,
                brier=brier_vals, nll=nll_vals, entropy=ent_vals, consistency=consistency_records)


def run_nl_tracker_episode(seed):
    rng = np.random.default_rng(seed)
    true_state = int(rng.integers(2))
    obs_idx = _step_env(rng, true_state, None)
    genie_belief = np.array([0.5, 0.5])
    action = "listen"
    belief_text = "BELIEF: no evidence yet, 50% left / 50% right."
    ret_disc = ret_undisc = 0.0
    listens = 0
    tokens = 0
    latency = 0.0
    calls = 0
    abstentions = 0
    malformed = 0
    brier_vals, nll_vals, ent_vals = [], [], []
    consistency_records = []

    for t in range(HORIZON):
        act_idx = MODEL.A.index(action)
        genie_belief = BELIEF_FILTER.update(genie_belief, act_idx, obs_idx)

        obs_name = "hear-left" if obs_idx == 0 else "hear-right"
        prompt = format_nl_prompt(belief_text, action, obs_name)
        action, meta = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, TEMPERATURE, PARSER, "listen",
                                 return_text=True)
        tokens += meta["prompt_tokens"] + meta["completion_tokens"]
        latency += meta["latency_s"]; calls += meta["calls"]; abstentions += int(meta["abstained"])

        belief_text = parse_belief_line(meta["text"]) if meta["text"] else belief_text
        parsed_belief, ok = parse_nl_belief(belief_text)
        if not ok:
            malformed += 1
        brier_vals.append(brier(parsed_belief, true_state))
        nll_vals.append(nll(parsed_belief, true_state))
        ent_vals.append(entropy(parsed_belief))
        consistency_records.append((tuple(np.round(genie_belief, 2)), prompt))

        if action == "listen":
            ret_disc += -1.0 * (GAMMA ** t); ret_undisc += -1.0; listens += 1
            obs_idx = _step_env(rng, true_state, obs_idx)
        else:
            won = (action == "open-right" and true_state == 0) or (action == "open-left" and true_state == 1)
            r = 10.0 if won else -100.0
            ret_disc += r * (GAMMA ** t); ret_undisc += r
            return dict(method="NL-Tracker", outcome="treasure" if won else "eaten", ret_disc=ret_disc,
                        ret_undisc=ret_undisc, listens=listens, steps=t + 1, tokens=tokens, latency=latency,
                        calls=calls, abstentions=abstentions, malformed=malformed, brier=brier_vals, nll=nll_vals,
                        entropy=ent_vals, consistency=consistency_records)

    return dict(method="NL-Tracker", outcome="timeout", ret_disc=ret_disc, ret_undisc=ret_undisc, listens=listens,
                steps=HORIZON, tokens=tokens, latency=latency, calls=calls, abstentions=abstentions,
                malformed=malformed, brier=brier_vals, nll=nll_vals, entropy=ent_vals, consistency=consistency_records)


def aggregate(method_name, episodes):
    rets = [e["ret_disc"] for e in episodes]
    mean, lo, hi = bootstrap_mean_ci(rets)
    outcomes = defaultdict(int)
    for e in episodes:
        outcomes[e["outcome"]] += 1
    all_brier = [b for e in episodes for b in e["brier"]]
    all_nll = [v for e in episodes for v in e["nll"]]
    all_ent = [v for e in episodes for v in e["entropy"]]
    return {
        "method": method_name,
        "n_episodes": len(episodes),
        "mean_discounted_return": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "std_discounted_return": float(np.std(rets)),
        "outcomes": dict(outcomes),
        "avg_listens": float(np.mean([e["listens"] for e in episodes])),
        "mean_brier": float(np.mean(all_brier)) if all_brier else None,
        "mean_nll": float(np.mean(all_nll)) if all_nll else None,
        "mean_entropy": float(np.mean(all_ent)) if all_ent else None,
        "avg_tokens_per_episode": float(np.mean([e["tokens"] for e in episodes])),
        "avg_latency_s_per_episode": float(np.mean([e["latency"] for e in episodes])),
        "avg_calls_per_episode": float(np.mean([e["calls"] for e in episodes])),
        "abstention_rate": float(sum(e["abstentions"] for e in episodes) / sum(e["calls"] for e in episodes)),
        "malformed_belief_rate": (float(sum(e.get("malformed", 0) for e in episodes) /
                                         sum(e["calls"] for e in episodes)) if "malformed" in episodes[0] else None),
    }


def measure_decision_consistency(method_name, episodes, policy_fn_builder):
    """Group steps by the rounded genie belief; where >=2 distinct prompts share
    a belief bucket, resample each prompt K times and report JSD across them."""
    groups = defaultdict(set)
    for e in episodes:
        for belief_key, prompt in e["consistency"]:
            groups[belief_key].add(prompt)
    candidate_groups = [g for g in groups.values() if len(g) >= 2]
    candidate_groups.sort(key=lambda g: -len(g))
    candidate_groups = candidate_groups[:N_JSD_GROUPS]

    jsd_values = []
    for group in candidate_groups:
        prompts = list(group)[:3]  # cap distinct prompts per group to bound cost
        dists = []
        for prompt in prompts:
            samples = []
            for _ in range(K_JSD_SAMPLES):
                action, _ = llm_call(CLIENT, SYSTEM_PROMPT, prompt, MODEL_NAME, TEMPERATURE, PARSER, "listen")
                samples.append(action)
            dists.append(action_distribution(samples, MODEL.A))
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


def run_method(method_name, episode_fn, n=N_MAIN):
    logger.info("Running %s (%d seeds, horizon=%d)...", method_name, n, HORIZON)
    episodes = [None] * n
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(episode_fn, seed): seed for seed in range(n)}
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
    for name, fn in [("Reactive", run_reactive_episode), ("BSE", run_bse_episode), ("NL-Tracker", run_nl_tracker_episode)]:
        eps = run_method(name, fn, N_MAIN)
        episode_cache[name] = eps
        results[name] = aggregate(name, eps)

    logger.info("Measuring decision consistency (JSD) for each method...")
    consistency = {}
    for name in ["Reactive", "BSE", "NL-Tracker"]:
        consistency[name] = measure_decision_consistency(name, episode_cache[name], None)

    out_path = os.path.join(LOG_DIR, "tiger_full_eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"config": {"n_main": N_MAIN, "horizon": HORIZON, "gamma": GAMMA, "temperature": TEMPERATURE,
                               "model": MODEL_NAME, "k_jsd_samples": K_JSD_SAMPLES, "n_jsd_groups": N_JSD_GROUPS},
                    "results": results, "consistency": consistency}, f, indent=2)
    logger.info("Saved results to %s", out_path)
    for name, r in results.items():
        logger.info("%s: %s", name, json.dumps(r))
    for name, c in consistency.items():
        logger.info("%s consistency: %s", name, json.dumps(c))
