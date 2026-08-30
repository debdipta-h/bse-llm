import numpy as np
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().with_name(".env"))

@dataclass
class POMDPModel:
    S: list; A: list; O: list
    T: np.ndarray; Z: np.ndarray; R: np.ndarray
    gamma: float; mu_0: np.ndarray

class BeliefFilter:
    def __init__(self, transition: np.ndarray, observation: np.ndarray):
        self.transition = transition
        self.observation = observation

    def update(self, b: np.ndarray, action_idx: int, obs_idx: int) -> np.ndarray:
        # Step 1: Prediction
        b_bar = np.dot(b, self.transition[action_idx])
        
        # Step 2: Correction
        b_tilde = b_bar * self.observation[action_idx, :, obs_idx]
        
        # Normalization
        eta = np.sum(b_tilde)
        if eta == 0:
            return np.ones_like(b) / len(b)
        return b_tilde / eta


class ExactMatchParser:
    def __init__(self, valid_actions: list[str]):
        self.valid_actions = set(valid_actions)

    def parse(self, response: str) -> str | None:
        for line in response.strip().splitlines():
            if line.startswith("ACTION:"):
                action = line.removeprefix("ACTION:").strip()
                if action in self.valid_actions:
                    return action
        return None

class LLMPolicy:
    def __init__(self, parser: ExactMatchParser, default_action: str):
        self.parser = parser
        self.default_action = default_action
        self.primary_temp = 0.3
        self.client = OpenAI()

    def get_action(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a precise agent. Output your reasoning, then exactly 'ACTION: <action>'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.primary_temp,
            )
            action = self.parser.parse(response.choices[0].message.content or "")
            if action is not None:
                return action
        except Exception:
            pass
        return self.default_action