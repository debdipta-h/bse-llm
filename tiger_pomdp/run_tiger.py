import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

# 1. Ensure Python can resolve core_engine from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiger_pomdp.environment import create_tiger_env
from core_engine import BeliefFilter, ExactMatchParser, LLMPolicy

# 2. Logging Setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TigerLiveExperiment")

def format_bse_prompt(belief, actions):
    return (
        "DOMAIN: Tiger POMDP\n"
        "GOAL: Choose the door with treasure while avoiding the tiger.\n"
        "REWARDS: +10 for treasure, -100 for tiger, -1 for listening.\n"
        f"AVAILABLE ACTIONS: {', '.join(actions)}\n\n"
        "CURRENT BELIEF POSTERIOR:\n"
        f"- tiger-left:  {belief[0]:.4f}\n"
        f"- tiger-right: {belief[1]:.4f}\n\n"
        "DECISION RULE:\n"
        "- Opening a door at 85% confidence yields a negative expected return (0.85*10 + 0.15*-100 = -6.5).\n"
        "- Listen again if confidence is below 95% to compound belief safely.\n"
        "- Only open a door when confidence is > 95% (e.g., tiger-left > 0.95 -> open-right).\n"
        "Output strictly as 'ACTION: <action>'."
    )

def format_reactive_prompt(obs_name, actions):
    return (
        "DOMAIN: Tiger POMDP\n"
        "GOAL: Choose the door without the tiger.\n"
        "REWARDS: +10 for treasure, -100 for tiger, -1 for listening.\n"
        f"AVAILABLE ACTIONS: {', '.join(actions)}\n\n"
        f"LATEST SENSOR OBSERVATION: {obs_name}\n\n"
        "INSTRUCTION: Output reasoning, followed by a final line strictly formatted as 'ACTION: <action>'."
    )


class BSEAugmentedAgent:
    def __init__(self):
        self.model = create_tiger_env()
        self.belief_filter = BeliefFilter(self.model.T, self.model.Z)
        self.llm_policy = LLMPolicy(ExactMatchParser(self.model.A), default_action="listen")

    def update_belief(self, belief, action, observation):
        return self.belief_filter.update(belief, self.model.A.index(action), observation)

    def get_action(self, belief):
        return self.llm_policy.get_action(format_bse_prompt(belief, self.model.A))


def run_tiger_visualization(num_seeds=50, horizon=20):
    if not os.environ.get("OPENAI_API_KEY"):
        logger.critical("FATAL: OPENAI_API_KEY environment variable is missing.")
        sys.exit(1)

    model = create_tiger_env()
    bse_filter = BeliefFilter(model.T, model.Z)
    parser = ExactMatchParser(model.A)
    llm_policy = LLMPolicy(parser, default_action="listen")

    results = {
        'Reactive': {'returns': [], 'eaten': 0, 'treasure': 0},
        'BSE':      {'returns': [], 'eaten': 0, 'treasure': 0}
    }

    logger.info(f"Starting Tiger POMDP Paired-Seed Protocol ({num_seeds} seeds) with LIVE GPT-4o...")

    for seed in range(num_seeds):
        logger.info(f"\n{'='*50}\nSTARTING SEED {seed}\n{'='*50}")
        np.random.seed(seed)
        true_state = np.random.choice([0, 1])

        # -------------------------------------------------------------
        # 1. Reactive LLM Baseline (GPT-4o conditioned only on observation)
        # -------------------------------------------------------------
        logger.info(">>> Running Reactive LLM <<<")
        obs_idx = true_state if np.random.rand() < 0.85 else (1 - true_state)
        ret_react = 0

        for t in range(horizon):
            obs_name = "hear-left" if obs_idx == 0 else "hear-right"
            prompt = format_reactive_prompt(obs_name, model.A)
            
            logger.info(f"[Reactive Turn {t+1}] Calling OpenAI with observation '{obs_name}'...")
            act = llm_policy.get_action(prompt)
            logger.info(f"[Reactive Turn {t+1}] GPT-4o Action: {act}")

            if act == 'listen':
                ret_react += -1.0 * (0.95 ** t)
                obs_idx = true_state if np.random.rand() < 0.85 else (1 - true_state)
            else:
                won = (act == 'open-right' and true_state == 0) or (act == 'open-left' and true_state == 1)
                ret_react += (10.0 if won else -100.0) * (0.95 ** t)
                if won:
                    results['Reactive']['treasure'] += 1
                    logger.info("Outcome: TREASURE FOUND (+10)")
                else:
                    results['Reactive']['eaten'] += 1
                    logger.info("Outcome: EATEN BY TIGER (-100)")
                break

        results['Reactive']['returns'].append(ret_react)

        # -------------------------------------------------------------
        # 2. BSE-Augmented LLM (GPT-4o conditioned on Bayes posterior)
        # -------------------------------------------------------------
        logger.info("\n>>> Running BSE-Augmented LLM <<<")
        np.random.seed(seed)
        obs_idx = true_state if np.random.rand() < 0.85 else (1 - true_state)
        belief = np.array([0.5, 0.5])
        ret_bse = 0
        act = "listen"

        for t in range(horizon):
            # Engine Bayes Update
            act_idx = model.A.index(act)
            belief = bse_filter.update(belief, act_idx, obs_idx)
            
            prompt = format_bse_prompt(belief, model.A)
            logger.info(f"[BSE Turn {t+1}] Calling OpenAI with posterior [L:{belief[0]:.4f}, R:{belief[1]:.4f}]...")
            act = llm_policy.get_action(prompt)
            logger.info(f"[BSE Turn {t+1}] GPT-4o Action: {act}")

            if act == 'listen':
                ret_bse += -1.0 * (0.95 ** t)
                obs_idx = true_state if np.random.rand() < 0.85 else (1 - true_state)
            else:
                won = (act == 'open-right' and true_state == 0) or (act == 'open-left' and true_state == 1)
                ret_bse += (10.0 if won else -100.0) * (0.95 ** t)
                if won:
                    results['BSE']['treasure'] += 1
                    logger.info("Outcome: TREASURE FOUND (+10)")
                else:
                    results['BSE']['eaten'] += 1
                    logger.info("Outcome: EATEN BY TIGER (-100)")
                break

        results['BSE']['returns'].append(ret_bse)

    # -------------------------------------------------------------
    # 3. Visualization
    # -------------------------------------------------------------
    logger.info("\nGenerating Visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot([results['Reactive']['returns'], results['BSE']['returns']], tick_labels=['Reactive LLM', 'BSE + LLM'])
    ax1.set_title('Discounted Returns (Tiger POMDP)')
    ax1.set_ylabel('Return')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    labels = ['Reactive LLM', 'BSE + LLM']
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, [results['Reactive']['treasure'], results['BSE']['treasure']], width, label='Treasure (+10)', color='green')
    ax2.bar(x + width/2, [results['Reactive']['eaten'], results['BSE']['eaten']], width, label='Eaten (-100)', color='red')
    ax2.set_title('Survival Outcomes')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    plt.tight_layout()
    output_path = 'tiger_results_live.png'
    plt.savefig(output_path)
    logger.info(f"Saved visualization to '{output_path}'")

if __name__ == "__main__":
    run_tiger_visualization(num_seeds=200, horizon=15)