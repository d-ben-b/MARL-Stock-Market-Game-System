def get_highway_config(style: str, duration: int = 40) -> dict:
    """
    根據指定的駕駛風格與時長，回傳 highway-v0 的環境配置。
    """
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
        "duration": duration,  # 動態接收外部設定的時間
    }

    if style == "conservative":
        config.update(
            {
                "collision_reward": -2.0,
                "reward_speed_range": [10, 20],
                "lane_change_reward": -0.5,
            }
        )
    elif style == "aggressive":
        config.update(
            {
                "collision_reward": -4.0,
                "reward_speed_range": [30, 40],
                "high_speed_reward": 2.0,
                "lane_change_reward": 0.2,
                "right_lane_reward": 0.0,
            }
        )
    else:
        config.update({"collision_reward": -1.0, "reward_speed_range": [20, 30]})

    return config
