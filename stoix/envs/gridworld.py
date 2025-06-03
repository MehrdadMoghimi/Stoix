import jax
import jax.numpy as jnp
import gymnax
from gymnax.environments import spaces  # Make sure this is gymnax.environments.spaces
from flax import struct
from typing import Tuple, Optional, Dict, Any
import chex
from functools import partial


# CellProperty remains the same
@struct.dataclass
class CellProperty:
    pos: Tuple[int, int]
    cumulant_fn: callable  # Takes a PRNGKey, returns float (this is the env's "reward")
    is_terminal: bool
    is_stochastic: bool


@struct.dataclass
class EnvParamsGW:  # GridWorld Params
    grid_size: Tuple[int, int] = (4, 4)
    # No gamma here as stock is managed by wrapper
    cell_properties: CellProperty = struct.field(
        default_factory=lambda: (
            CellProperty(pos=(3, 0), cumulant_fn=lambda key: -1.0, is_terminal=False, is_stochastic=False),  # Red
            CellProperty(pos=(0, 3), cumulant_fn=lambda key: 2.0, is_terminal=False, is_stochastic=False),  # Red
            # CellProperty(
            #    pos=(0, 3), cumulant_fn=lambda key: -2.0 * jax.random.bernoulli(key, 0.5).astype(jnp.float32), is_terminal=False, is_stochastic=True
            # ),  # Yellow
            # CellProperty(
            #    pos=(2, 2), cumulant_fn=lambda key: 3.0 * jax.random.bernoulli(key, 0.5).astype(jnp.float32), is_terminal=True, is_stochastic=True
            # ),  # Gray (3B, T)
            CellProperty(pos=(3, 3), cumulant_fn=lambda key: 0.0, is_terminal=True, is_stochastic=False),  # Gray (0, T)
        )
    )
    initial_agent_pos: Tuple[int, int] = (0, 0)
    max_steps_in_episode: int = 16


@struct.dataclass
class EnvStateGW:  # GridWorld State
    agent_pos: chex.Array  # (row, col)
    key: chex.PRNGKey  # Key for stochastic rewards from cells
    time: int


class GridWorldEnv(gymnax.environments.environment.Environment[EnvStateGW, EnvParamsGW]):
    """
    GridWorld environment with a grid of cells, each with its own reward function.
    The agent can move in four directions (up, down, left, right) and receives
    rewards based on the cell it enters. The environment is episodic and has a
    maximum number of steps per episode.
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParamsGW:
        return EnvParamsGW()

    # @partial(jax.jit, static_argnames=("self",))
    def reset(self, key: chex.PRNGKey, params: EnvParamsGW) -> Tuple[chex.Array, EnvStateGW]:
        """Performs resetting of environment."""
        return self.reset_env(key, params)

    # @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvStateGW,
        action: int,
        params: EnvParamsGW,
    ) -> Tuple[chex.Array, EnvStateGW, float, bool, Dict]:
        """Performs step transition of environment."""
        return self.step_env(key, state, action, params)

    def step_env(self, key: chex.PRNGKey, state: EnvStateGW, action: int, params: EnvParamsGW) -> Tuple[chex.Array, EnvStateGW, float, bool, Dict]:
        # key_move is not strictly needed if actions are deterministic,
        # but good practice if state.key is for cell rewards
        key_cell_reward = state.key  # Use the key from the state for cell rewards

        is_currently_terminal = jnp.zeros((), dtype=jnp.bool_)
        for cell_prop in params.cell_properties:
            is_currently_terminal |= jnp.array_equal(state.agent_pos, jnp.array(cell_prop.pos)) & cell_prop.is_terminal

        def absorbed_step(current_state):
            obs = self._get_obs(current_state, params)
            # No movement, 0 reward (cumulant), done
            return obs, current_state, 0.0, True, {}

        def active_step(current_state, action_val, key_for_cells):
            # UP, DOWN, LEFT, RIGHT, STAY
            # 0,    1,     2,     3,      4
            d_row = jnp.array([-1, 1, 0, 0, 0])
            d_col = jnp.array([0, 0, -1, 1, 0])

            new_row = current_state.agent_pos[0] + d_row[action_val]
            new_col = current_state.agent_pos[1] + d_col[action_val]

            new_row = jnp.clip(new_row, 0, params.grid_size[0] - 1)
            new_col = jnp.clip(new_col, 0, params.grid_size[1] - 1)
            new_agent_pos = jnp.array([new_row, new_col])

            reward_from_env = 0.0  # This is the cumulant
            done = jnp.zeros((), dtype=jnp.bool_)

            # Split the key for stochastic cell rewards
            # One subkey per cell property that might be stochastic
            num_stochastic_cells = len(params.cell_properties)  # Max possible needed
            keys_for_cell_rewards = jax.random.split(key_for_cells, num_stochastic_cells + 1)
            next_state_key = keys_for_cell_rewards[0]  # Key for the next state

            current_subkey_idx = 0
            for i, cell_prop in enumerate(params.cell_properties):
                is_target_cell = jnp.array_equal(new_agent_pos, jnp.array(cell_prop.pos))
                # Use a unique pre-split key for each cell's cumulant function
                reward_from_this_cell = cell_prop.cumulant_fn(keys_for_cell_rewards[i + 1])

                reward_from_env = jnp.where(is_target_cell, reward_from_this_cell, reward_from_env)
                done |= is_target_cell & cell_prop.is_terminal

            new_time = current_state.time + 1
            done |= new_time >= params.max_steps_in_episode

            next_env_state = EnvStateGW(agent_pos=new_agent_pos, key=next_state_key, time=new_time)  # Pass the new key for the next step
            obs = self._get_obs(next_env_state, params)
            # The "reward" returned by this env IS the cumulant R_t+1
            return obs, next_env_state, reward_from_env, done, {}

        # The input 'key' to step_env is now used to derive the next state's key
        # if an active step is taken.
        return jax.lax.cond(is_currently_terminal, absorbed_step, lambda s: active_step(s, action, key), state)  # Pass the main step key here

    def reset_env(self, key: chex.PRNGKey, params: EnvParamsGW) -> Tuple[chex.Array, EnvStateGW]:
        # The key passed to reset becomes the initial key in the state
        state = EnvStateGW(agent_pos=jnp.array(params.initial_agent_pos), key=key, time=0)
        obs = self._get_obs(state, params)
        return obs, state

    def _get_obs(self, state: EnvStateGW, params: EnvParamsGW) -> chex.Array:  # Returns just the image
        grid = jnp.zeros(params.grid_size, dtype=jnp.float32)
        grid = grid.at[state.agent_pos[0], state.agent_pos[1]].set(1.0)
        return grid.reshape((*params.grid_size, 1))  # Observation is just the image

    def observation_space(self, params: EnvParamsGW) -> spaces.Box:  # Obs is now just Box
        return spaces.Box(0, 1, (*params.grid_size, 1), jnp.float32)

    def action_space(self, params: EnvParamsGW) -> spaces.Discrete:
        return spaces.Discrete(5)

    @property
    def name(self) -> str:
        return "GridWorld-v0"

    @property
    def num_actions(self) -> int:
        return 5
