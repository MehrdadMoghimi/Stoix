"""HIVTreatment environment."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from jax.experimental.ode import odeint


@struct.dataclass
class EnvState(environment.EnvState):
    patient_state: jnp.ndarray  # Your 6 state variables (T1, T2, T1s, T2s, V, E)
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):

    min_obs: float = -5.0
    max_obs: float = 8.0
    dt: float = 20  #: measurement every 20 days
    logspace: bool = True  #: whether observed states are in log10 space or not --> ALWAYS TRUE
    dosage_noise: float = 0.15
    max_noise: float = 0.3
    max_eps1: float = 0.7
    max_eps2: float = 0.3
    max_steps_in_episode: int = 50


class HIVTreatmentEnv(environment.Environment[EnvState, EnvParams]):
    """
    Simulation of HIV Treatment. The aim is to find an optimal drug schedule.

    **STATE:** The state contains concentrations of 6 different cells:

    * T1: non-infected CD4+ T-lymphocytes [cells / ml]
    * T1*:    infected CD4+ T-lymphocytes [cells / ml]
    * T2: non-infected macrophages [cells / ml]
    * T2*:    infected macrophages [cells / ml]
    * V: number of free HI viruses [copies / ml]
    * E: number of cytotoxic T-lymphocytes [cells / ml]

    **ACTIONS:** The therapy consists of 2 drugs
    (reverse transcriptase inhibitor [RTI] and protease inhibitor [PI]) which
    are activated or not. The action space contains therefore of 4 actions:

    * *0*: none active
    * *1*: RTI active
    * *2*: PI active
    * *3*: RTI and PI active

    **REFERENCE:**

    .. seealso::
        Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
        Clinical data based optimal STI strategies for HIV:
        A reinforcement learning approach
        In Proceedings of the 45th IEEE Conference on Decision and Control (2006).


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
        patient_state = state.patient_state

        eps1, eps2 = action[0], action[1]

        # rescale to action space
        eps1 = (eps1 + 1) / 2 * params.max_eps1
        eps2 = (eps2 + 1) / 2 * params.max_eps1

        # scale by noise level
        eps1 = eps1 * (1 + params.dosage_noise * jax.random.normal(key))
        eps2 = eps2 * (1 + params.dosage_noise * jax.random.normal(key))

        # clip
        eps1 = jnp.clip(eps1, 0.0, (1 + params.max_noise) * params.max_eps1)
        eps2 = jnp.clip(eps2, 0.0, (1 + params.max_noise) * params.max_eps2)

        def dsdt(s, t, eps1, eps2):
            """
            system derivate per time. The unit of time are days.
            """
            # model parameter constants
            lambda1 = 1e4
            lambda2 = 31.98
            d1 = 0.01
            d2 = 0.01
            f = 0.34
            k1 = 8e-7
            k2 = 1e-4
            delta = 0.7
            m1 = 1e-5
            m2 = 1e-5
            NT = 100.0
            c = 13.0
            rho1 = 1.0
            rho2 = 1.0
            lambdaE = 1
            bE = 0.3
            Kb = 100
            d_E = 0.25
            Kd = 500
            deltaE = 0.1

            # decompose state
            T1, T2, T1s, T2s, V, E = s

            # compute derivatives
            tmp1 = (1.0 - eps1) * k1 * V * T1
            tmp2 = (1.0 - f * eps1) * k2 * V * T2
            dT1 = lambda1 - d1 * T1 - tmp1
            dT2 = lambda2 - d2 * T2 - tmp2
            dT1s = tmp1 - delta * T1s - m1 * E * T1s
            dT2s = tmp2 - delta * T2s - m2 * E * T2s
            dV = (1.0 - eps2) * NT * delta * (T1s + T2s) - c * V - ((1.0 - eps1) * rho1 * k1 * T1 + (1.0 - f * eps1) * rho2 * k2 * T2) * V
            dE = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

            return jnp.array([dT1, dT2, dT1s, dT2s, dV, dE])

        ns = odeint(dsdt, patient_state, jnp.array([0.0, params.dt]), eps1, eps2, mxstep=1000)[-1]
        T1, T2, T1s, T2s, V, E = ns
        new_patient_state = jnp.array([T1, T2, T1s, T2s, V, E])
        reward = -0.1 * V - 2e4 * eps1**2 - 2e3 * eps2**2 + 1e3 * E
        reward = reward / 1e6 - 1.0
        reward = reward.squeeze()

        # Update state dict and evaluate termination conditions
        state = EnvState(
            patient_state=new_patient_state,
            time=state.time + 1,
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
        init_state = jnp.array([163573.0, 5.0, 11945.0, 46.0, 63919.0, 24.0])
        state = EnvState(patient_state=init_state, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        return jnp.log10(state.patient_state).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done.squeeze()

    @property
    def name(self) -> str:
        """Environment name."""
        return "HIVTreatment-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=0,
            high=1,
            shape=(2,),
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(params.min_obs, params.max_obs, shape=(6,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "patient_state": spaces.Box(params.min_obs, params.max_obs, (6,), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
