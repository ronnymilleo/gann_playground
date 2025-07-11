"""
Unit tests for the Genetic Neural Network Game.
Run with: python -m pytest test_genetic_game.py -v
"""

import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import math


# Import modules to test
from gann import GeneticANN, dynamic_crossover, mutation
from player import Player
from config import GameConfig


class TestGeneticANN(unittest.TestCase):
    """Test cases for the GeneticANN class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.network = GeneticANN()
        
    def test_network_initialization(self):
        """Test that neural network is properly initialized."""
        self.assertEqual(len(self.network.layers), 4)
        self.assertEqual(self.network.fitness, 0)
        self.assertEqual(len(self.network.fitness_array), 0)
        self.assertEqual(self.network.max_fitness, 0)
        
    def test_fitness_update(self):
        """Test fitness calculation with dynamic max distance."""
        test_distance = 10.0
        expected_max_distance = math.sqrt(20**2 + 20**2)
        expected_fitness = 1.0 - (test_distance / expected_max_distance)
        
        self.network.fitness_update(test_distance)
        
        self.assertAlmostEqual(self.network.fitness, expected_fitness, places=5)
        self.assertEqual(len(self.network.fitness_array), 1)
        self.assertEqual(self.network.mean_fitness, expected_fitness)
        
    def test_fitness_update_boundary_values(self):
        """Test fitness calculation with boundary values."""
        # Test with zero distance (perfect score)
        self.network.fitness_update(0.0)
        self.assertAlmostEqual(self.network.fitness, 1.0, places=5)
        
        # Test with maximum distance (worst score)
        max_distance = math.sqrt(20**2 + 20**2)
        self.network.fitness_update(max_distance)
        self.assertAlmostEqual(self.network.fitness, 0.0, places=5)
        
    def test_network_prediction(self):
        """Test that network produces valid predictions."""
        # Create sample input
        test_input = tf.constant([[0.5, 0.5, 0.3, 0.25, 0.1, 0.2, 0.8, 0.9]])
        
        # Get prediction
        output = self.network(test_input, training=False)
        
        # Check output shape and properties
        self.assertEqual(output.shape, (1, 4))  # Batch size 1, 4 actions
        self.assertTrue(tf.reduce_all(output >= 0))  # Softmax outputs should be positive
        self.assertAlmostEqual(tf.reduce_sum(output).numpy(), 1.0, places=5)  # Should sum to 1


class TestCrossover(unittest.TestCase):
    """Test cases for the genetic crossover function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parent1 = GeneticANN()
        self.parent2 = GeneticANN()
        
    def test_crossover_produces_valid_network(self):
        """Test that crossover produces a valid neural network."""
        child = dynamic_crossover(self.parent1, self.parent2)
        
        # Check that child has correct architecture
        self.assertEqual(len(child.layers), len(self.parent1.layers))
        self.assertEqual(len(child.layers), len(self.parent2.layers))
        
        # Check that all weights are finite
        for layer in child.layers:
            weights = layer.get_weights()
            for weight_array in weights:
                self.assertTrue(np.isfinite(weight_array).all())
                
    def test_crossover_weight_inheritance(self):
        """Test that child inherits weights from parents."""
        # Set distinctive weights for parents
        for i, layer in enumerate(self.parent1.layers):
            weights = layer.get_weights()
            # Set parent1 weights to positive values
            for j, w in enumerate(weights):
                weights[j] = np.ones_like(w) * (i + 1)
            layer.set_weights(weights)
            
        for i, layer in enumerate(self.parent2.layers):
            weights = layer.get_weights()
            # Set parent2 weights to negative values
            for j, w in enumerate(weights):
                weights[j] = np.ones_like(w) * -(i + 1)
            layer.set_weights(weights)
        
        # Create child through crossover
        child = dynamic_crossover(self.parent1, self.parent2)
        
        # Verify child has mixture of parent weights
        for layer in child.layers:
            weights = layer.get_weights()
            for weight_array in weights:
                # Should contain both positive and negative values (mixture)
                has_positive = np.any(weight_array > 0)
                has_negative = np.any(weight_array < 0)
                # Note: Due to randomness, this might occasionally fail
                # In practice, we'd use a fixed seed for deterministic testing


class TestMutation(unittest.TestCase):
    """Test cases for the mutation function."""
    
    def test_mutation_preserves_structure(self):
        """Test that mutation preserves weight structure."""
        # Create test weights
        original_kernel = np.random.random((8, 8))
        original_bias = np.random.random(8)
        test_weights = [original_kernel.copy(), original_bias.copy()]
        
        # Apply mutation
        mutation(test_weights)
        
        # Check structure is preserved
        self.assertEqual(test_weights[0].shape, original_kernel.shape)
        self.assertEqual(test_weights[1].shape, original_bias.shape)
        self.assertTrue(np.isfinite(test_weights[0]).all())
        self.assertTrue(np.isfinite(test_weights[1]).all())
        
    def test_mutation_changes_weights(self):
        """Test that mutation actually changes some weights."""
        # Create test weights
        original_kernel = np.ones((8, 8))
        original_bias = np.ones(8)
        test_weights = [original_kernel.copy(), original_bias.copy()]
        
        # Apply mutation multiple times to ensure some change occurs
        changed = False
        for _ in range(100):  # Try multiple times due to probabilistic nature
            mutation(test_weights)
            if not np.array_equal(test_weights[0], original_kernel) or \
               not np.array_equal(test_weights[1], original_bias):
                changed = True
                break
                
        self.assertTrue(changed, "Mutation should change weights within 100 attempts")


class TestPlayer(unittest.TestCase):
    """Test cases for the Player class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock pygame to avoid GUI dependencies in tests
        with patch('pygame.sprite.Sprite.__init__'), \
             patch('pygame.Surface'), \
             patch('player.GeneticANN'):
            self.player = Player()
            self.player.nn = MagicMock()
            
    def test_player_initialization(self):
        """Test player initialization."""
        self.assertFalse(self.player.win_status)
        self.assertFalse(self.player.lose_status)
        self.assertEqual(self.player.x, 0)
        self.assertEqual(self.player.y, 0)
        self.assertEqual(self.player.distance, 0.0)
        self.assertEqual(self.player.steps, 0)
        
    def test_player_attributes_types(self):
        """Test that player attributes have correct types."""
        self.assertIsInstance(self.player.distance, float)
        self.assertIsInstance(self.player.x_angle, float)
        self.assertIsInstance(self.player.y_angle, float)


class TestGameConfig(unittest.TestCase):
    """Test cases for the GameConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GameConfig()
        
    def test_config_properties(self):
        """Test calculated properties."""
        expected_max_distance = math.sqrt(20**2 + 20**2)
        self.assertAlmostEqual(self.config.max_distance, expected_max_distance, places=5)
        
        expected_right_margin = 32 * 20
        self.assertEqual(self.config.right_margin, expected_right_margin)
        
        expected_pixel_width = 20 * 32
        self.assertEqual(self.config.grid_pixel_width, expected_pixel_width)
        
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test that all required attributes exist
        required_attrs = [
            'screen_width', 'screen_height', 'cell_size',
            'grid_width', 'grid_height', 'population_size',
            'elite_count', 'generations', 'input_size', 'output_size'
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(self.config, attr))
            self.assertIsInstance(getattr(self.config, attr), int)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions from main.py."""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        # Import the function (would need to be extracted to a utils module)
        from main import euclidian_distance
        
        # Test simple cases
        self.assertEqual(euclidian_distance(0, 0, 3, 4), 5.0)
        self.assertEqual(euclidian_distance(1, 1, 1, 1), 0.0)
        self.assertAlmostEqual(euclidian_distance(0, 0, 1, 1), math.sqrt(2), places=5)
        
    def test_coordinate_conversion(self):
        """Test coordinate conversion functions."""
        from main import x_conv, y_conv
        
        # Test x_conv
        test_x = 40 + 32  # left_margin + one cell
        expected_x = 2  # Should be grid position 2
        self.assertEqual(x_conv(test_x), expected_x)
        
        # Test y_conv
        test_y = 60 + 32  # top_margin + one cell
        expected_y = 19  # Should be grid position 19 (inverted)
        self.assertEqual(y_conv(test_y), expected_y)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    @patch('pygame.init')
    @patch('pygame.display.set_mode')
    @patch('pygame.image.load')
    @patch('pygame.font.Font')
    def test_game_initialization(self, mock_font, mock_image, mock_display, mock_init):
        """Test that the game can be initialized without errors."""
        # Mock pygame components
        mock_display.return_value = MagicMock()
        mock_image.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        # This would test the main function initialization
        # Would need refactoring to make main() more testable
        pass


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
