import torch
from abc import ABC, abstractmethod
import random

class BasePlanner(ABC):
    @abstractmethod
    def generate_subgoal(self, state_description: str) -> str:
        pass

class MockPlanner(BasePlanner):
    """
    Deterministic mock planner for testing and CPU environments.
    """
    def __init__(self, env_id="MiniGrid-DoorKey-6x6-v0"):
        self.env_id = env_id
        # Simple finite state machine logic for specific envs could go here
        # or just a mapping based on keywords in the state description.

    def generate_subgoal(self, state_description: str) -> str:
        """
        Returns a canonical subgoal string based on heuristics.
        """
        desc = state_description.lower()

        # Heuristic for DoorKey
        # 1. If carrying nothing and see key -> Pick up key
        # 2. If carrying key and see door -> Open door
        # 3. If door is open -> Go to goal

        # Correctly check for carrying key vs carrying nothing
        is_carrying_key = "carrying" in desc and "key" in desc and "carrying nothing" not in desc
        is_carrying_nothing = "carrying nothing" in desc

        if is_carrying_key:
            if "closed" in desc and "door" in desc:
                # Find color of door if possible, else generic
                return "Open the door"
            else:
                return "Go to the goal"

        if "key" in desc and is_carrying_nothing:
            return "Pick up the key"

        if "goal" in desc:
             return "Go to the goal"

        return "Explore"

class Phi2Planner(BasePlanner):
    def __init__(self, model_name="microsoft/phi-2", load_in_4bit=True, use_lora=False, device_map="auto", temperature=0.1, max_new_tokens=50):
        if not torch.cuda.is_available():
             raise RuntimeError("Phi2Planner requires CUDA. Use MockPlanner instead.")

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )

        if use_lora:
            # Placeholder for loading adapters
            pass

    def generate_subgoal(self, state_description: str) -> str:
        # Improved prompt with few-shot examples and rules
        prompt = (
            "Objective: Reach the goal.\n"
            "Rules:\n"
            "1. If you see a key and are carrying nothing, pick up the key.\n"
            "2. If you have the key and see a door, open the door.\n"
            "3. If the door is open, go to the goal.\n\n"
            "Example 1:\n"
            "Current State: You are carrying nothing. In the room, you see: yellow key, yellow door, green goal.\n"
            "Next Subgoal: Pick up the yellow key\n\n"
            "Example 2:\n"
            "Current State: You are carrying a yellow key. In the room, you see: yellow door, green goal.\n"
            "Next Subgoal: Open the yellow door\n\n"
            f"Current State: {state_description}\n"
            "Next Subgoal:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the part after "Next Subgoal:"
        if "Next Subgoal:" in text:
            subgoal = text.split("Next Subgoal:")[-1].strip().split("\n")[0]
        else:
            subgoal = text.strip()

        return subgoal

def get_planner(config):
    """
    Factory function to return the appropriate planner.
    """
    mock_mode = config['llm'].get('mock_mode', 'auto')

    use_mock = False
    if mock_mode == True:
        use_mock = True
    elif mock_mode == 'auto':
        if not torch.cuda.is_available():
            use_mock = True

    if use_mock:
        print("Using MockPlanner (CPU/Testing Mode)")
        return MockPlanner(env_id=config['env']['id'])
    else:
        print("Using Phi2Planner (GPU Mode)")
        return Phi2Planner(
            model_name=config['llm']['model_name'],
            load_in_4bit=config['llm']['load_in_4bit'],
            use_lora=config['llm']['use_lora'],
            temperature=config['llm']['temperature'],
            max_new_tokens=config['llm']['max_new_tokens']
        )
