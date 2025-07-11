"""
Configuration management for the Genetic Neural Network Game.
This module provides centralized configuration for all game parameters.
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GameConfig:
    """Central configuration for the genetic algorithm game."""
    
    # Display settings
    screen_width: int = 1440
    screen_height: int = 900
    cell_size: int = 32
    
    # Grid settings
    grid_width: int = 20
    grid_height: int = 20
    left_margin: int = 40
    top_margin: int = 60
    down_margin: int = 20
    
    # Population settings
    population_size: int = 50
    elite_count: int = 5
    generations: int = 100
    max_steps_per_generation: int = 30
    
    # Neural network architecture
    input_size: int = 8
    hidden_layer_1_size: int = 8
    hidden_layer_2_size: int = 8
    hidden_layer_3_size: int = 6
    output_size: int = 4
    
    # Genetic algorithm parameters
    mutation_rate: float = 0.4
    crossover_probability: float = 0.5
    
    # Environment settings
    min_start_target_distance: int = 5  # Minimum grid cells between start and target
    objective_change_frequency: int = 10  # Change start/target every N generations
    
    # Colors (RGB)
    background_color: Tuple[int, int, int] = (255, 255, 255)
    player_active_color: Tuple[int, int, int] = (51, 51, 230)
    player_success_color: Tuple[int, int, int] = (51, 230, 51)
    player_failed_color: Tuple[int, int, int] = (230, 51, 51)
    
    @property
    def right_margin(self) -> int:
        """Calculate right margin based on grid size."""
        return self.cell_size * self.grid_width
    
    @property
    def max_distance(self) -> float:
        """Calculate maximum possible distance on the grid."""
        return math.sqrt(self.grid_width**2 + self.grid_height**2)
    
    @property
    def grid_pixel_width(self) -> int:
        """Total pixel width of the game grid."""
        return self.grid_width * self.cell_size
    
    @property
    def grid_pixel_height(self) -> int:
        """Total pixel height of the game grid."""
        return self.grid_height * self.cell_size


# Global configuration instance
config = GameConfig()


def update_config(**kwargs) -> None:
    """Update configuration parameters dynamically."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")


def get_config() -> GameConfig:
    """Get the current configuration instance."""
    return config


# Legacy support - maintain backward compatibility with globals.py
width = config.screen_width
height = config.screen_height
b_size = config.cell_size
left_margin = config.left_margin
right_margin = config.right_margin
top_margin = config.top_margin
down_margin = config.down_margin
generations = config.generations
population = config.population_size
PLAYER_RED = config.player_failed_color
PLAYER_GREEN = config.player_success_color
PLAYER_BLUE = config.player_active_color
WHITE = config.background_color
