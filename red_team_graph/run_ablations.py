import os
import sys
import logging
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from .environment import create_k6_attack_graph
    from .run_attack import BSEGraphAgent
except ImportError:
    from environment import create_k6_attack_graph
    from run_attack import BSEGraphAgent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs", date.today().isoformat())
os.makedirs(EXPERIMENT_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(EXPERIMENT_LOG_DIR, "graph_ablations.log")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="w")],
)
logger = logging.getLogger("GraphAblations")

def run_graph_ablation_case(ablation_name, num_seeds=3, horizon=20):
    model = create_k6_attack_graph()
    agent = BSEGraphAgent(model.S, model.A)
    
    if ablation_name == "AB9_High_Temp":
        agent.llm_policy.primary_temp = 1.0 
    else:
        agent.llm_policy.primary_temp = 0.3

    returns = []
    
    logger.info("--- Running Graph Ablation: %s ---", ablation_name)
    for seed in range(num_seeds):
        np.random.seed(seed)
        true_state = np.random.choice([0, 1], size=6)
        
        belief = np.ones(64) / 64.0
        ret_case = 0
        action = "wait"
        obs = np.random.choice([0, 1])
        
        for t in range(horizon):
            act_idx = model.A.index(action)
            
            # 1. Apply the mathematical ablations
            if ablation_name == "AB1_No_Predict":
                b_tilde = belief * model.Z[act_idx, :, obs]
                belief = b_tilde / np.sum(b_tilde)
            elif ablation_name == "AB2_No_Observe":
                belief = np.dot(belief, model.T[act_idx])
            else:
                # Standard 2-Step Bayes Filter
                b_bar = np.dot(belief, model.T[act_idx])
                b_tilde = b_bar * model.Z[act_idx, :, obs]
                belief = b_tilde / np.sum(b_tilde)

            # 2. Get Live LLM Action
            action = agent.get_action(belief)
            
            # 3. Step Environment (Simplified physical step)
            if 'scan' in action:
                target = int(action.split('_')[1])
                is_vuln = true_state[target] == 1
                obs = 1 if (is_vuln and np.random.rand() < 0.85) or (not is_vuln and np.random.rand() < 0.10) else 0
                ret_case += -0.5 * (0.95 ** t)
            elif 'exploit' in action:
                target = int(action.split('_')[1])
                ret_case += (20 if true_state[target] == 1 else -5) * (0.95 ** t)
                break
            else:
                ret_case += -1 * (0.95 ** t)
                
        returns.append(ret_case)
        logger.info("Completed seed %s/%s for %s", seed + 1, num_seeds, ablation_name)
        
    return returns

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        logger.critical("FATAL: OPENAI_API_KEY is missing. Add it to .env before running.")
        sys.exit(1)

    res_std = run_graph_ablation_case("Standard_BSE")
    res_ab1 = run_graph_ablation_case("AB1_No_Predict")
    res_ab2 = run_graph_ablation_case("AB2_No_Observe")
    res_ab9 = run_graph_ablation_case("AB9_High_Temp")

    plt.figure(figsize=(10, 6))
    plt.boxplot([res_std, res_ab1, res_ab2, res_ab9], 
                tick_labels=['Standard BSE', 'AB1\n(No Predict)', 'AB2\n(No Observe)', 'AB9\n(Temp 1.0)'])
    plt.title('Attack Graph Ablation Study: Component Impact', fontsize=14)
    plt.ylabel('Discounted Return')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('graph_ablations_live.png')
    logger.info("Ablation graph saved to graph_ablations_live.png")