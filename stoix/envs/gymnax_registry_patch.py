"""
Patch for the Gymnax registration system to allow custom environments.
This must be imported before using any custom environments.
"""

import importlib
from typing import Dict, Type, Any

import gymnax
from gymnax.environments.environment import Environment

# Convert the list to a dictionary for better lookup
# Store the original list for compatibility
_original_registered_envs = gymnax.registration.registered_envs
_env_registry: Dict[str, Type[Environment]] = {}

# Register the original environments
for env_id in _original_registered_envs:
    _env_registry[env_id] = None  # Placeholder, will be created by original make()

# Replace the original make function
_original_make = gymnax.registration.make


def _patched_make(env_id: str, **env_kwargs):
    """Patched version of gymnax.make that supports custom environments."""
    # If it's a custom environment in our registry with a class
    if env_id in _env_registry and _env_registry[env_id] is not None:
        env_class = _env_registry[env_id]
        env = env_class(**env_kwargs)
        return env, env.default_params

    # Otherwise, fall back to the original make function
    return _original_make(env_id, **env_kwargs)


def register_env(env_id: str, env_class: Type[Environment]) -> None:
    """Register a custom environment with Gymnax."""
    if env_id in _env_registry:
        raise ValueError(f"Environment ID '{env_id}' is already registered")

    # Add to our registry
    _env_registry[env_id] = env_class

    # Update the Gymnax registered_envs list
    if env_id not in gymnax.registration.registered_envs:
        gymnax.registration.registered_envs.append(env_id)

    # print(f"Registered custom environment: {env_id}")


# Apply the patch
gymnax.registration.make = _patched_make
gymnax.make = _patched_make

# Export the register function
__all__ = ["register_env"]
