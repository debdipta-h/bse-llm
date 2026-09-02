"""
Shared statistics and instrumented-LLM-call helpers used by the run_full_eval.py
scripts in tiger_pomdp/ and red_team_graph/.

Implements the four metric families defined in the paper (Section VI.C):
task return (bootstrap CI), belief calibration (Brier / NLL / entropy),
decision consistency (Jensen-Shannon divergence over belief-equivalent
history pairs), and compute cost (tokens / latency / abstentions).
"""
import time
import numpy as np

EPS = 1e-12


def bootstrap_mean_ci(data, n_resamples=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI on the mean of `data`."""
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = data[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(data.mean()), float(lo), float(hi)


def brier(belief, true_idx):
    belief = np.asarray(belief, dtype=float)
    target = np.zeros_like(belief)
    target[true_idx] = 1.0
    return float(np.sum((belief - target) ** 2))


def nll(belief, true_idx):
    belief = np.asarray(belief, dtype=float)
    return float(-np.log(max(belief[true_idx], EPS)))


def entropy(belief):
    belief = np.asarray(belief, dtype=float)
    p = np.clip(belief, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def jsd(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    def kl(a, b):
        a = np.clip(a, EPS, 1.0)
        b = np.clip(b, EPS, 1.0)
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def action_distribution(samples, actions):
    counts = np.array([samples.count(a) for a in actions], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.ones(len(actions)) / len(actions)
    return counts / total


def llm_call(client, system_prompt, user_prompt, model_name, temperature, parser, default_action, return_text=False):
    """
    Stateless, thread-safe single decision call with the same fallback
    protocol as core_engine.LLMPolicy: one resample at temperature 0.0 on a
    malformed response, then a deterministic default action.
    Returns (action: str, meta: dict) where meta carries token/latency/
    abstention instrumentation for the compute-cost metric family.
    If return_text is True, meta also carries the raw completion under "text"
    (needed by methods that must read more than the ACTION line, e.g. the
    NL-Tracker's self-reported belief).
    """
    meta = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
            "calls": 0, "abstained": False, "text": ""}

    def _one_call(temp):
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
        )
        meta["latency_s"] += time.perf_counter() - t0
        meta["calls"] += 1
        if response.usage is not None:
            meta["prompt_tokens"] += response.usage.prompt_tokens or 0
            meta["completion_tokens"] += response.usage.completion_tokens or 0
        text = response.choices[0].message.content or ""
        if return_text:
            meta["text"] = text
        return parser.parse(text)

    try:
        action = _one_call(temperature)
        if action is not None:
            return action, meta
        action = _one_call(0.0)
        if action is not None:
            return action, meta
    except Exception:
        pass

    meta["abstained"] = True
    return default_action, meta
