# Per-environment speed targets (m/s). Roundabout/intersection run at lower speeds.
_SPEED_RANGES: dict[str, dict[str, list[int]]] = {
    "highway-v0":      {"base": [20, 30], "conservative": [10, 20], "aggressive": [30, 40]},
    "merge-v0":        {"base": [20, 30], "conservative": [10, 20], "aggressive": [25, 35]},
    "roundabout-v0":   {"base": [5,  15], "conservative": [3,  10], "aggressive": [10, 20]},
    "intersection-v0": {"base": [5,  15], "conservative": [3,  10], "aggressive": [10, 20]},
}


def get_env_config(env_id: str = "highway-v0", style: str = "base", duration: int = 40) -> dict:
    """Return highway-env config for any supported environment and driving style."""
    speed_table = _SPEED_RANGES.get(env_id, _SPEED_RANGES["highway-v0"])
    speed_range = speed_table.get(style, speed_table["base"])

    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },
        "action": {"type": "DiscreteMetaAction"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": duration,
        "reward_speed_range": speed_range,
    }

    if style == "conservative":
        config.update({
            "collision_reward": -2.0,
            "lane_change_reward": -0.5,
        })
    elif style == "aggressive":
        config.update({
            "collision_reward": -4.0,
            "high_speed_reward": 2.0,
            "lane_change_reward": 0.2,
            "right_lane_reward": 0.0,
        })
    else:
        config.update({"collision_reward": -1.0})

    return config


def get_highway_config(style: str, duration: int = 40) -> dict:
    """Backward-compatible wrapper — delegates to get_env_config."""
    return get_env_config("highway-v0", style, duration)
