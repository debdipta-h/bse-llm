import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Link to the root core_engine.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from .environment import TigerEnvironment
    from .run_tiger import BSEAugmentedAgent
except ImportError:
    from environment import TigerEnvironment
    from run_tiger import BSEAugmentedAgent

def run_tiger_ablation_case(ablation_name, num_seeds=3, horizon=15):
    env = TigerEnvironment()
    
    # Initialize the live GPT-4o agent
    # Assuming BSEAugmentedAgent in run_tiger.py uses USE_REAL_LLM=True
    agent = BSEAugmentedAgent() 
    
    # AB9: Temperature Sweep
    if ablation_name == "AB9_High_Temp":
        agent.llm_policy.primary_temp = 1.0 
    else:
        agent.llm_policy.primary_temp = 0.3

    returns = []
    
    print(f"\n--- Running Ablation: {ablation_name} ---")
    for seed in range(num_seeds):
        obs = env.reset(seed)
        belief = np.array([0.5, 0.5])
        ret_case = 0
        action = "listen"
        
        for t in range(horizon):
            # 1. Apply the specific architectural ablation to the math
            if ablation_name == "AB1_No_Predict":
                # Skip Transition matrix. Apply observation correction directly.
                new_b = np.zeros(2)
                if obs == 0:
                    new_b[0], new_b[1] = belief[0] * 0.85, belief[1] * 0.15
                else:
                    new_b[0], new_b[1] = belief[0] * 0.15, belief[1] * 0.85
                belief = new_b / np.sum(new_b)
                
            elif ablation_name == "AB2_No_Observe":
                # Skip Observation matrix. Only predict transitions.
                if action != 'listen':
                    belief = np.array([0.5, 0.5])
                # Do not multiply by 0.85 or 0.15
                
            else:
                # Standard BSE (Algorithm 1)
                belief = agent.update_belief(belief, action, obs)

            # 2. Get Live LLM Action
            action = agent.get_action(belief)
            
            # 3. Step Environment
            obs, reward, done, outcome = env.step(action)
            ret_case += reward * (0.95 ** t)
            
            if done: 
                break
                
        returns.append(ret_case)
        print(f"Completed Seed {seed+1}/{num_seeds} for {ablation_name}")
        
    return returns

if __name__ == "__main__":
    # Ensure API Key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY is missing. Export it before running.")
        sys.exit(1)

    # Run the cases
    res_std = run_tiger_ablation_case("Standard_BSE")
    res_ab1 = run_tiger_ablation_case("AB1_No_Predict")
    res_ab2 = run_tiger_ablation_case("AB2_No_Observe")
    res_ab9 = run_tiger_ablation_case("AB9_High_Temp")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.boxplot([res_std, res_ab1, res_ab2, res_ab9], 
                tick_labels=['Standard BSE', 'AB1\n(No Predict)', 'AB2\n(No Observe)', 'AB9\n(Temp 1.0)'])
    plt.title('Tiger POMDP Ablation Study: Impact of Removing BSE Components', fontsize=14)
    plt.ylabel('Discounted Return')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    filename = 'tiger_ablations_live.png'
    plt.savefig(filename)
    print(f"\nAblation graph saved to {filename}")