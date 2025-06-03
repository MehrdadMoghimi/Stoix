"""Trading environment."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState:
    stock_price: jnp.ndarray
    agent_inventory: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams:
    min_action: float = -2.0
    max_action: float = 2.0
    min_q: float = -5.0  # minimum value for the inventory
    max_q: float = 5.0  # maximum value for the inventory
    kappa: float = 2.0  # kappa of the OU process
    sigma: float = 1.0  # standard deviation of the OU process or volatility of the GBM process
    theta: float = 1.0  # mean-reversion level of the OU process or initial price of the GBM process
    phi: float = 0.005  # transaction costs
    psi: float = 0.5  # terminal penalty on the inventory
    T: float = 1.0  # trading horizon
    Ndt: float = 10.0  # number of periods


class TradingEnv(environment.Environment[EnvState, EnvParams]):
    """
    JAX version of TradingEnv
    """

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # reward is calculated with the current stock price and the current action
        # Only is time = T-1, the reward also includes the terminal penalty on the inventory
        reward = state.stock_price * action - params.phi * jnp.power(action, 2)

        # price of the stock at next time step - OU process
        dt = params.T / params.Ndt
        eta = params.sigma * jnp.sqrt((1 - jnp.exp(-2 * params.kappa * dt)) / (2 * params.kappa))
        stock_price = (
            params.theta
            + (state.stock_price - params.theta) * jnp.exp(-params.kappa * dt)
            + eta * jax.random.normal(key)
        )

        time_step = state.time + 1
        # inventory at next time step - add the trade to current inventory
        agent_inventory = state.agent_inventory + action

        done = self.is_terminal(state, params)  # Check if the state.time_step is NdT
        # reward - profit with terminal penalty calculated with the new price of the stock and the new inventory
        reward += (
            agent_inventory * stock_price - params.psi * jnp.power(agent_inventory, 2)
        ) * done
        reward = reward.squeeze()

        # Update state dict and evaluate termination conditions
        state = EnvState(
            jnp.array(stock_price).squeeze(),
            jnp.array(agent_inventory).squeeze(),
            time_step,
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        state = EnvState(
            stock_price=jnp.array(params.theta), agent_inventory=jnp.array(0.0), time=0
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.stock_price, state.agent_inventory, state.time]).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.Ndt
        return done.squeeze()

    @property
    def name(self) -> str:
        """Environment name."""
        return "TradingEnv-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(1,),
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            [
                params.theta - 6 * params.sigma / jnp.sqrt(2 * params.kappa),
                params.min_q,
                0.0,
            ],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [
                params.theta + 6 * params.sigma / jnp.sqrt(2 * params.kappa),
                params.max_q,
                params.Ndt,
            ],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (3,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "stock_price": spaces.Box(
                    params.theta - 6 * params.sigma / jnp.sqrt(2 * params.kappa),
                    params.theta + 6 * params.sigma / jnp.sqrt(2 * params.kappa),
                    (),
                    jnp.float32,
                ),
                "agent_inventory": spaces.Box(params.min_q, params.max_q, (), jnp.float32),
                "time": spaces.Discrete(params.Ndt),
            }
        )
