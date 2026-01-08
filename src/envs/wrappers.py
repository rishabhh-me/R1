import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX

class SubgoalWrapper(gym.Wrapper):
    """
    Wraps MiniGrid env to:
    1. Return Dict observation (image + subgoal).
    2. Provide textual description of state.
    3. Detect subgoal completion (placeholder logic).
    """
    def __init__(self, env):
        super().__init__(env)

        # Current subgoal (tuple and ID)
        self.current_subgoal_tuple = ("no_op", "none", "none")
        self.current_subgoal_id = 0

        # Define Observation Space
        # SB3 handles Dict spaces. We need 'image' (original) and 'subgoal' (scalar or one-hot).
        # We'll use a scalar for now (easier for Embedding layer).
        self.observation_space = spaces.Dict({
            "image": env.observation_space["image"], # (H, W, 3)
            "subgoal": spaces.Box(low=0, high=30, shape=(1,), dtype=np.int32) # Assuming < 30 subgoals
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # self.current_subgoal_id = 0 # Reset to NO_OP -> We want to persist subgoal for training phase if set
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if subgoal is complete (This needs external feedback or internal logic)
        # For Phase 0, we assume the Planner/Agent loop handles "when to switch subgoal"
        # or we check it here. Let's provide a 'subgoal_completed' flag in info.
        is_complete = self._check_subgoal_completion(obs)
        info['subgoal_completed'] = is_complete

        if is_complete:
            reward += 1.0  # Intrinsic reward for completing the subgoal

        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, base_obs):
        return {
            "image": base_obs["image"],
            "subgoal": np.array([self.current_subgoal_id], dtype=np.int32)
        }

    def set_subgoal(self, subgoal_tuple, subgoal_id):
        """
        Called by the external loop to update the current subgoal.
        """
        self.current_subgoal_tuple = subgoal_tuple
        self.current_subgoal_id = subgoal_id

    def get_text_description(self):
        """
        Returns a textual description of the current state for the LLM.
        """
        # This relies on the fact that we can access the underlying MiniGrid env
        # Note: In a wrapper stack, we might need to access `self.unwrapped`
        grid = self.unwrapped.grid
        agent_pos = self.unwrapped.agent_pos
        agent_dir = self.unwrapped.agent_dir

        # Simple local view or global view?
        # Let's do a simple scan of what is visible or what is in the room.

        desc = []

        # Check what the agent is carrying
        carrying = self.unwrapped.carrying
        if carrying:
            desc.append(f"You are carrying a {carrying.color} {carrying.type}.")
        else:
            desc.append("You are carrying nothing.")

        # Check for objects in front or nearby (Simplified)
        # We can iterate over the grid to find key objects.

        found_items = []
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell and cell.type in ['key', 'door', 'goal', 'ball', 'box']:
                    # Simple relative position or just existence
                    found_items.append(f"{cell.color} {cell.type}")

        if found_items:
            desc.append("In the room, you see: " + ", ".join(found_items) + ".")

        # Add basic info about door state if visible
        # (This is a bit cheaty, accessing global state, but fine for Phase 0 sanity)

        return " ".join(desc)

    def _check_subgoal_completion(self, obs):
        """
        Heuristic check if current subgoal is satisfied.
        """
        action_type, color, obj_type = self.current_subgoal_tuple

        if action_type == "pick":
             # Check if carrying the object
             carrying = self.unwrapped.carrying
             if carrying and carrying.type == obj_type:
                 # Color check
                 if color == "any" or carrying.color == color:
                     return True

        elif action_type == "open":
            # Check if door is open. Harder to check from just 'carrying'.
            # Need to check grid state.
            # Assuming there is only one door for Phase 0 sanity check usually.
            for i in range(self.unwrapped.grid.width):
                for j in range(self.unwrapped.grid.height):
                    cell = self.unwrapped.grid.get(i, j)
                    if cell and cell.type == 'door' and cell.is_open:
                        return True

        elif action_type == "goto":
            # Check proximity? Or just let the PPO agent maximize reward?
            # For "Go to goal", standard reward signal handles it.
            pass

        return False
