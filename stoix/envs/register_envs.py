"""Register custom environments with Gymnax."""

# Import the patch first
from stoix.envs.gymnax_registry_patch import register_env

# Import your custom environments
from stoix.envs.hivtreatment import HIVTreatmentEnv
from stoix.envs.trading import TradingEnv
from stoix.envs.gridworld import GridWorldEnv


def register_custom_environments():
    """Register all custom environments."""
    # Register your environment
    register_env("HIVTreatment-v0", HIVTreatmentEnv)
    register_env("Trading-v0", TradingEnv)
    register_env("GridWorld-v0", GridWorldEnv)

    # Add more environments as needed
    # register_env("AnotherEnv-v0", AnotherEnvClass)

    return True


# Register environments when this module is imported
_registered = register_custom_environments()
