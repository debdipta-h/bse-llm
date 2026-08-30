import os
import sys
import logging
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

# 1. PATH FIX: Tell Python to look in the root folder for core_engine.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .environment import create_k6_attack_graph
except ImportError:
    from environment import create_k6_attack_graph

from core_engine import BeliefFilter, ExactMatchParser, LLMPolicy

# 2. LOGGING SETUP (console and date-stamped experiment log file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs", date.today().isoformat())
os.makedirs(EXPERIMENT_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(EXPERIMENT_LOG_DIR, "attack_graph_run.log")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="w")],
)
logger = logging.getLogger("RealGraphExperiment")

def format_bse_prompt(belief_state, states, actions):
    """
    Computes marginal vulnerability probabilities for each node from the 64-state belief.
    """
    K = 6
    node_probs = []
    for node_idx in range(K):
        # Sum belief mass over all states where node_idx is vulnerable (state[node_idx] == 1)
        prob_vuln = sum(belief_state[s_idx] for s_idx, state in enumerate(states) if state[node_idx] == 1)
        node_probs.append(prob_vuln)
        
    prompt = (
        "DOMAIN: K=6 Red-Team Attack Graph\n"
        "GOAL: Discover vulnerable entry points and execute successful exploits.\n"
        "REWARDS: +20 for successful exploit, -5 for failed exploit, -0.5 for scan, -1 for wait.\n"
        f"AVAILABLE ACTIONS: {', '.join(actions)}\n\n"
        "CURRENT MARGINAL PROBABILITY OF NODES BEING VULNERABLE:\n"
    )
    
    for i, p in enumerate(node_probs):
        confidence = "High Confidence Vulnerable" if p > 0.85 else ("High Confidence Hardened" if p < 0.15 else "Uncertain")
        prompt += f"- Node {i}: {p:.4f} ({confidence})\n"
        
    prompt += (
        "\nSTRATEGY INSTRUCTION:\n"
        "- If a node has > 85% vulnerability probability, exploit it (e.g., 'ACTION: exploit_0').\n"
        "- If uncertain, scan the node to gather intelligence (e.g., 'ACTION: scan_0').\n"
        "- Avoid exploiting nodes with low confidence to prevent the -5 penalty.\n"
    )
    return prompt

def format_reactive_prompt(obs, actions):
    return (
        "DOMAIN: K=6 Red-Team Attack Graph\n"
        "REWARDS: +20 for successful exploit, -5 for failed exploit, -0.5 for scanning.\n"
        f"AVAILABLE ACTIONS: {', '.join(actions)}\n\n"
        f"LATEST SCAN OBSERVATION: {obs}\n"
    )


class BSEGraphAgent:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.llm_policy = LLMPolicy(ExactMatchParser(actions), default_action="scan_0")

    def get_action(self, belief_state):
        prompt = format_bse_prompt(belief_state, self.states, self.actions)
        return self.llm_policy.get_action(prompt)


def run_real_attack_visualization(num_seeds=3, horizon=15):
    logger.info("Initializing K=6 Attack Graph with LIVE GPT-4o...")
    model = create_k6_attack_graph()
    
    # Initialize Engine Components
    bse_filter = BeliefFilter(model.T, model.Z)
    parser = ExactMatchParser(model.A)
    llm_policy = LLMPolicy(parser, default_action="scan_0")
    
    results = {'Reactive': {'returns': [], 'compromised': 0, 'failed': 0},
               'BSE':      {'returns': [], 'compromised': 0, 'failed': 0}}
    
    for seed in range(num_seeds):
        logger.info(f"\n{'='*50}\nSTARTING SEED {seed}\n{'='*50}")
        np.random.seed(seed)
        true_state = np.random.choice([0, 1], size=6)
        
        # --- 1. REACTIVE AGENT (GPT-4o without Belief) ---
        logger.info(">>> Running Reactive LLM <<<")
        obs = np.random.choice([0, 1])
        ret_react = 0
        
        for t in range(horizon):
            prompt = format_reactive_prompt(obs, model.A)
            logger.info(f"[Reactive Turn {t+1}] Calling OpenAI...")
            
            action = llm_policy.get_action(prompt)
            logger.info(f"[Reactive Turn {t+1}] GPT-4o Decided: {action}")
            
            # Step Environment
            if 'scan' in action:
                target = int(action.split('_')[1])
                is_vuln = true_state[target] == 1
                obs = 1 if (is_vuln and np.random.rand() < 0.85) or (not is_vuln and np.random.rand() < 0.10) else 0
                ret_react += -0.5 * (0.95 ** t)
            elif 'exploit' in action:
                target = int(action.split('_')[1])
                if true_state[target] == 1:
                    ret_react += 20 * (0.95 ** t)
                    results['Reactive']['compromised'] += 1
                    logger.info("Outcome: NETWORK COMPROMISED (+20)")
                else:
                    ret_react += -5 * (0.95 ** t)
                    results['Reactive']['failed'] += 1
                    logger.info("Outcome: EXPLOIT FAILED (-5)")
                break 
            else:
                ret_react += -1 * (0.95 ** t)
        results['Reactive']['returns'].append(ret_react)

        # --- 2. BSE AGENT (GPT-4o WITH Belief) ---
        logger.info("\n>>> Running BSE-Augmented LLM <<<")
        belief = np.ones(64) / 64.0
        ret_bse = 0
        action = "wait"
        obs = np.random.choice([0, 1])
        
        for t in range(horizon):
            # 1. Update Math
            act_idx = model.A.index(action)
            belief = bse_filter.update(belief, act_idx, obs)
            
            # 2. Format Prompt & Call LLM
            prompt = format_bse_prompt(belief, model.S,model.A)
            logger.info(f"[BSE Turn {t+1}] Calling OpenAI with updated Posterior...")
            
            action = llm_policy.get_action(prompt)
            logger.info(f"[BSE Turn {t+1}] GPT-4o Decided: {action}")
            
            # 3. Step Environment
            if 'scan' in action:
                target = int(action.split('_')[1])
                is_vuln = true_state[target] == 1
                obs = 1 if (is_vuln and np.random.rand() < 0.85) or (not is_vuln and np.random.rand() < 0.10) else 0
                ret_bse += -0.5 * (0.95 ** t)
            elif 'exploit' in action:
                target = int(action.split('_')[1])
                if true_state[target] == 1:
                    ret_bse += 20 * (0.95 ** t)
                    results['BSE']['compromised'] += 1
                    logger.info("Outcome: NETWORK COMPROMISED (+20)")
                else:
                    ret_bse += -5 * (0.95 ** t)
                    results['BSE']['failed'] += 1
                    logger.info("Outcome: EXPLOIT FAILED (-5)")
                break
            else:
                ret_bse += -1 * (0.95 ** t)
                
        results['BSE']['returns'].append(ret_bse)

    # --- PLOTTING ---
    logger.info("\nGenerating Visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.boxplot([results['Reactive']['returns'], results['BSE']['returns']], tick_labels=['Reactive LLM', 'BSE + LLM'])
    ax1.set_title('Discounted Returns (Live GPT-4o)')
    ax1.set_ylabel('Return')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    labels = ['Reactive LLM', 'BSE + LLM']
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, [results['Reactive']['compromised'], results['BSE']['compromised']], width, label='Compromised (+20)', color='green')
    ax2.bar(x + width/2, [results['Reactive']['failed'], results['BSE']['failed']], width, label='Failed Exploit (-5)', color='red')
    ax2.set_title('Network Intrusion Outcomes')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('attack_graph_real_llm.png')
    logger.info("Saved visualization to 'attack_graph_real_llm.png'")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        logger.critical("FATAL: OPENAI_API_KEY missing. Export it before running.")
        sys.exit(1)
        
    run_real_attack_visualization(num_seeds=600)