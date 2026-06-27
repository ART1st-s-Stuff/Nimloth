"""AI2-THOR navigation environment manager.

Wraps VAGEN's ``NavigationService`` with a clean Nimloth-native interface,
including action-index mapping and trajectory recording.
"""

from nimloth.environment.navigation.manager import NavigationEnvManager

__all__ = [
    "NavigationEnvManager",
]
