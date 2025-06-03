from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.specs import Array, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from stoix.base_types import Observation, StockObservation, StockEnvState
from stoix.wrappers.gymnax import gymnax_space_to_jumanji_spec


class FlattenObservationWrapper(Wrapper):
    """Simple wrapper that flattens the agent view observation."""

    def __init__(self, env: Environment) -> None:
        self._env = env
        obs_shape = self._env.observation_spec().agent_view.shape
        self._obs_shape = (np.prod(obs_shape),)

    def _flatten(self, obs: Observation) -> Array:
        agent_view = obs.agent_view.astype(jnp.float32)
        return agent_view.reshape(self._obs_shape)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        agent_view = self._flatten(timestep.observation)
        timestep = timestep.replace(
            observation=timestep.observation._replace(agent_view=agent_view),
        )
        if "next_obs" in timestep.extras:
            agent_view = self._flatten(timestep.extras["next_obs"])
            timestep.extras["next_obs"] = timestep.extras["next_obs"]._replace(
                agent_view=agent_view
            )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        agent_view = self._flatten(timestep.observation)
        timestep = timestep.replace(
            observation=timestep.observation._replace(agent_view=agent_view),
        )
        if "next_obs" in timestep.extras:
            agent_view = self._flatten(timestep.extras["next_obs"])
            timestep.extras["next_obs"] = timestep.extras["next_obs"]._replace(
                agent_view=agent_view
            )
        return state, timestep

    def observation_spec(self) -> Spec:
        return self._env.observation_spec().replace(
            agent_view=Array(shape=self._obs_shape, dtype=jnp.float32)
        )


class MultiDiscreteToDiscrete(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._action_spec_num_values = env.action_spec().num_values

    def apply_factorisation(self, x: chex.Array) -> chex.Array:
        """Applies the factorisation to the given action."""
        action_components = []
        flat_action = x
        n = self._action_spec_num_values.shape[0]
        for i in range(n - 1, 0, -1):
            flat_action, remainder = jnp.divmod(flat_action, self._action_spec_num_values[i])
            action_components.append(remainder)
        action_components.append(flat_action)
        action = jnp.stack(
            list(reversed(action_components)),
            axis=-1,
            dtype=self._action_spec_num_values.dtype,
        )
        return action

    def inverse_factorisation(self, y: chex.Array) -> chex.Array:
        """Inverts the factorisation of the given action."""
        n = self._action_spec_num_values.shape[0]
        action_components = jnp.split(y, n, axis=-1)
        flat_action = action_components[0]
        for i in range(1, n):
            flat_action = self._action_spec_num_values[i] * flat_action + action_components[i]
        return flat_action

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        action = self.apply_factorisation(action)
        state, timestep = self._env.step(state, action)
        return state, timestep

    def action_spec(self) -> specs.Spec:
        """Returns the action spec of the environment."""
        original_action_spec = self._env.action_spec()
        num_actions = int(np.prod(np.asarray(original_action_spec.num_values)))
        return specs.DiscreteArray(num_actions, name="action")


class MultiBoundedToBounded(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._true_action_shape = env.action_spec().shape

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        action = action.reshape(self._true_action_shape)
        state, timestep = self._env.step(state, action)
        return state, timestep

    def action_spec(self) -> specs.Spec:
        """Returns the action spec of the environment."""
        original_action_spec = self._env.action_spec()
        size = int(np.prod(np.asarray(original_action_spec.shape)))
        return specs.BoundedArray(
            (size,),
            minimum=original_action_spec.minimum,
            maximum=original_action_spec.maximum,
            dtype=original_action_spec.dtype,
            name="action",
        )


class AddStartFlagAndPrevAction(Wrapper):
    """Wrapper that adds a start flag and the previous action to the observation."""

    def __init__(self, env: Environment):
        super().__init__(env)

        # Get the action dimension
        if isinstance(self.action_spec(), specs.DiscreteArray):
            self.action_dim = self.action_spec().num_values
            self.discrete = True
        else:
            self.action_dim = self.action_spec().shape[0]
            self.discrete = False

        # Check if the observation is flat
        if not len(self.observation_spec().agent_view.shape) == 1:
            raise ValueError("The observation must be flat.")

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        state, timestep = self._env.reset(key)
        start_flag = jnp.array(1.0)[jnp.newaxis]
        prev_action = jnp.zeros(self.action_dim)
        agent_view = timestep.observation.agent_view
        new_agent_view = jnp.concatenate([start_flag, prev_action, agent_view])
        timestep = timestep.replace(
            observation=timestep.observation._replace(
                agent_view=new_agent_view,
            )
        )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        state, timestep = self._env.step(state, action)
        start_flag = jnp.array(0.0)[jnp.newaxis]
        prev_action = action
        if self.discrete:
            prev_action = jax.nn.one_hot(prev_action, self.action_dim)
        agent_view = timestep.observation.agent_view
        new_agent_view = jnp.concatenate([start_flag, prev_action, agent_view])
        timestep = timestep.replace(
            observation=timestep.observation._replace(
                agent_view=new_agent_view,
            )
        )
        return state, timestep

    def observation_spec(self) -> Spec:
        return self._env.observation_spec().replace(
            agent_view=Array(
                shape=(1 + self.action_dim + self._env.observation_spec().agent_view.shape[0],),
                dtype=jnp.float32,
            )
        )


class MakeChannelLast(Wrapper):
    """Simple wrapper for observations that have the channel dim first.
    This makes the channel dim last."""

    def __init__(self, env: Environment) -> None:
        self._env = env
        obs_shape = jnp.array(self._env.observation_spec().agent_view.shape)
        self._obs_shape = jnp.roll(obs_shape, len(obs_shape) - 1)

        assert len(self._obs_shape) > 2, "for > 2 dimensional observations"

    def _make_channel_last(self, obs: Observation) -> Array:
        agent_view = obs.agent_view
        agent_view = jnp.rollaxis(agent_view, 0, len(self._obs_shape))
        return agent_view

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        agent_view = self._make_channel_last(timestep.observation)
        timestep = timestep.replace(
            observation=timestep.observation._replace(agent_view=agent_view),
        )
        if "next_obs" in timestep.extras:
            agent_view = self._make_channel_last(timestep.extras["next_obs"])
            timestep.extras["next_obs"] = timestep.extras["next_obs"]._replace(
                agent_view=agent_view
            )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        agent_view = self._make_channel_last(timestep.observation)
        timestep = timestep.replace(
            observation=timestep.observation._replace(agent_view=agent_view),
        )
        if "next_obs" in timestep.extras:
            agent_view = self._make_channel_last(timestep.extras["next_obs"])
            timestep.extras["next_obs"] = timestep.extras["next_obs"]._replace(
                agent_view=agent_view
            )
        return state, timestep

    def observation_spec(self) -> Spec:
        return self._env.observation_spec().replace(
            agent_view=Array(shape=self._obs_shape, dtype=jnp.float32)
        )


class AddStock(Wrapper):
    """Wrapper that adds a stock."""

    def __init__(
        self,
        env: Environment,
        gamma: float,
        initial_stock: float = None,
        initial_stock_min: float = None,
        initial_stock_max: float = None,
    ):
        super().__init__(env)
        self._env = env
        self._gamma = gamma
        self._initial_stock = initial_stock
        self._initial_stock_min = initial_stock_min
        self._initial_stock_max = initial_stock_max

        # Get the action dimension
        if isinstance(self._env.action_spec(), specs.DiscreteArray):
            self.action_dim = self._env.action_spec().num_values
            self.discrete = True
        else:
            self.action_dim = self._env.action_spec().shape[0]
            self.discrete = False

        # check if whether initial_stock is not None or initial_stock_min and initial_stock_max are not None
        if initial_stock is None and (initial_stock_min is None or initial_stock_max is None):
            raise ValueError(
                "Either initial_stock or initial_stock_min and initial_stock_max must be provided."
            )

    def reset(self, key: chex.PRNGKey) -> Tuple[StockEnvState, TimeStep[StockObservation]]:

        key_env_reset, key_stock_init = jax.random.split(key)

        env_state, timestep = self._env.reset(key_env_reset)

        # If initial_stock is provided, use it; otherwise, generate a random initial stock
        if self._initial_stock is not None:
            initial_stock = jnp.array(self._initial_stock, dtype=jnp.float32)
        else:
            initial_stock = jax.random.uniform(
                key_stock_init,
                minval=self._initial_stock_min,
                maxval=self._initial_stock_max,
                dtype=jnp.float32,
                shape=(),
            )

        state = StockEnvState(key=key, env_state=env_state, stock=initial_stock)

        timestep = timestep.replace(
            observation=StockObservation(
                agent_view=timestep.observation.agent_view,
                action_mask=timestep.observation.action_mask,
                step_count=jnp.array(0, dtype=int),
                stock=initial_stock,
            )
        )
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[StockEnvState, TimeStep[StockObservation]]:
        env_state, timestep = self._env.step(state.env_state, action)
        new_stock = (state.stock + timestep.reward) / self._gamma

        state = StockEnvState(key=state.key, env_state=env_state, stock=new_stock)
        timestep = timestep.replace(
            observation=StockObservation(
                agent_view=timestep.observation.agent_view,
                action_mask=timestep.observation.action_mask,
                step_count=jnp.array(0, dtype=int),
                stock=new_stock,
            )
        )
        return state, timestep

    def observation_spec(self) -> Spec:
        agent_view_spec = gymnax_space_to_jumanji_spec(
            self._env.observation_space(self._env_params)
        )

        action_mask_spec = Array(shape=self._legal_action_mask.shape, dtype=float)

        return specs.Spec(
            StockObservation,
            "StockObservationSpec",
            agent_view=agent_view_spec,
            action_mask=action_mask_spec,
            stock=Array(shape=(), dtype=float),
            step_count=Array(shape=(), dtype=int),
        )
