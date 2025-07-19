import math
import os
import pathlib
import random
from os.path import join
from typing import List, Tuple, Optional

import numpy as np
import pygame
import tensorflow as tf
import numpy.typing as npt
# Import pygame.locals for easier access to key coordinates
from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

import gann
from globals import *
from player import Player

import logging
import time
from datetime import datetime

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genetic_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def validate_player_position(player: Player) -> bool:
    """Validate that player position is within grid bounds"""
    if not (0 <= player.x <= 20 and 0 <= player.y <= 20):
        print(f"Warning: Player position out of bounds: ({player.x}, {player.y})")
        return False
    return True


def validate_target_position(target_x: int, target_y: int) -> bool:
    """Validate that target position is within grid bounds"""
    grid_x, grid_y = x_conv(target_x), y_conv(target_y)
    if not (0 <= grid_x <= 20 and 0 <= grid_y <= 20):
        print(f"Warning: Target position out of bounds: ({grid_x}, {grid_y})")
        return False
    return True


# Drawing functions
def draw_target(screen, icon):
    screen.blit(icon, (t_px_pos_x, t_px_pos_y))


def draw_flag(screen, icon):
    screen.blit(icon, (f_px_pos_x, f_px_pos_y))


def draw_grid(screen):
    for x in range(left_margin, left_margin + right_margin + b_size, b_size):
        pygame.draw.line(screen, (0, 0, 0), (x, top_margin), (x, height - down_margin), 1)
    for y in range(top_margin, height - down_margin + b_size, b_size):
        pygame.draw.line(screen, (0, 0, 0), (left_margin, y), (left_margin + right_margin, y), 1)


def draw_text(screen, font, generation):
    gen = font.render("Generation = {}".format(generation), True, [0, 0, 0], [255, 255, 255])
    pop = font.render("Population = {}".format(population), True, [0, 0, 0], [255, 255, 255])
    steps = font.render("Steps = {}".format(step), True, [0, 0, 0], [255, 255, 255])
    screen.blits([(gen, (20, 20)), (pop, (20 + 150, 20)), (steps, (20 + 300, 20))])


def draw_fit_table(screen, font, players):
    player_text = font.render("Player", True, [0, 0, 0], [255, 255, 255])
    fitness_text = font.render("Fitness", True, [0, 0, 0], [255, 255, 255])
    steps_text = font.render("Steps", True, [0, 0, 0], [255, 255, 255])
    screen.blits([(player_text, (720, 60)),
                  (fitness_text, (720 + 100, 60)),
                  (steps_text, (720 + 200, 60))])
    half_players = len(players) // 2
    for p in range(0, len(players)):
        player_id = font.render("P{}".format(p), True, [0, 0, 0], [255, 255, 255])
        fitness_number = font.render("{:.4f}".format(players[p].nn.fitness), True, [0, 0, 0],
                                     [255, 255, 255])
        steps_number = font.render("{:02d}".format(players[p].steps), True, [0, 0, 0], [255, 255, 255])
        if p < half_players:
            screen.blits([(player_id, (720, 60 + p * 24 + 24)),
                          (fitness_number, (720 + 100, 60 + p * 24 + 24)),
                          (steps_number, (720 + 200, 60 + p * 24 + 24))])
        else:
            screen.blits([(player_id, (1020, 60 + (p - half_players) * 24 + 24)),
                          (fitness_number, (1020 + 100, 60 + (p - half_players) * 24 + 24)),
                          (steps_number, (1020 + 200, 60 + (p - half_players) * 24 + 24))])


def draw_players(screen, player: Player):
    screen.blit(player.image, player.rect)


def render(screen, f_icon, t_icon):
    draw_grid(screen)
    draw_flag(screen, f_icon)
    draw_target(screen, t_icon)
    pygame.display.flip()


def update_text(screen, font, generation, players):
    draw_text(screen, font, generation)
    draw_fit_table(screen, font, players)


# Calculations and converting functions
def euclidian_distance(x1, y1, x2, y2):
    new_distance = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return new_distance


def x_conv(px_x: int):
    return (px_x - left_margin) // 32 + 1


def y_conv(px_y: int):
    return 20 - (px_y - top_margin) // 32


def random_position():
    return math.floor(random.random() * (640 / b_size)) * b_size + left_margin, \
           math.floor(random.random() * (640 / b_size)) * b_size + top_margin


def move(p: Player):
    # Simplified and more effective inputs
    delta_x = (x_conv(t_px_pos_x) - p.x) / 20.0  # Normalized delta_x
    delta_y = (y_conv(t_px_pos_y) - p.y) / 20.0  # Normalized delta_y
    
    nn_input = [p.x / 20.0, 
                p.y / 20.0, 
                delta_x, 
                delta_y]
    
    input_tensor = tf.convert_to_tensor([nn_input], dtype=tf.float32)
    p.nn_output = p.nn(input_tensor, training=False)
    
    # Add safety check for neural network output
    if p.nn_output is None:
        # Default to random movement if network fails
        action = np.random.randint(0, 4)
    else:
        action = np.argmax(p.nn_output)
    
    # Execute movement based on action
    if action == 0:
        p.rect.move_ip(b_size, 0)
    elif action == 1:
        p.rect.move_ip(-b_size, 0)
    elif action == 2:
        p.rect.move_ip(0, b_size)
    else:
        p.rect.move_ip(0, -b_size)

    # Keep player on the screen and check lose status
    # Check boundaries BEFORE correcting position to avoid inconsistency
    if (p.rect.left <= left_margin or 
        p.rect.right >= b_size * 20 + left_margin or
        p.rect.top <= top_margin or 
        p.rect.bottom >= b_size * 20 + top_margin):
        p.lose_status = True
    
    # Then correct position after lose status is set
    if p.rect.left <= left_margin:
        p.rect.left = left_margin
    if p.rect.right >= b_size * 20 + left_margin:
        p.rect.right = b_size * 20 + left_margin
    if p.rect.top <= top_margin:
        p.rect.top = top_margin
    if p.rect.bottom >= b_size * 20 + top_margin:
        p.rect.bottom = b_size * 20 + top_margin

    # Check win status immediately after movement
    if p.rect.left == t_px_pos_x and p.rect.top == t_px_pos_y:
        p.win_status = True
        # Stop further movement for this player
        return





def update_player(p: Player, steps):
    # Only update if player is still active
    if p.lose_status or p.win_status:
        return
        
    # Update player info every round
    p.x = x_conv(p.rect.left)
    p.y = y_conv(p.rect.top)
    # Update distance based on movement
    p.distance = euclidian_distance(x_conv(t_px_pos_x),
                                    y_conv(t_px_pos_y),
                                    p.x,
                                    p.y)
    p.steps = steps
    # Update fitness only for active players
    p.nn.fitness_update(p.distance, p.steps)
    # Save max fitness info
    if p.nn.fitness > p.nn.max_fitness:
        p.nn.max_fitness = p.nn.fitness


def reset_players(players_array):
    # Update player's pixel position, distance to target and status
    for player in players_array:
        player.lose_status = False
        player.win_status = False
        player.x, player.y = x_conv(f_px_pos_x), y_conv(f_px_pos_y)
        player.rect.left, player.rect.top = f_px_pos_x, f_px_pos_y
        player.distance = euclidian_distance(x_conv(t_px_pos_x),
                                             y_conv(t_px_pos_y),
                                             player.x,
                                             player.y)
        player.steps = 0  # Reset step counter


def randomize_objectives():
    global f_px_pos_x, f_px_pos_y, t_px_pos_x, t_px_pos_y
    # Generate new random positions for start and target
    f_px_pos_x, f_px_pos_y = random_position()
    t_px_pos_x, t_px_pos_y = random_position()
    # Check if target is at same position as start flag
    while abs(t_px_pos_x - f_px_pos_x) < b_size * 5 and abs(t_px_pos_y - f_px_pos_y) < b_size * 5:
        t_px_pos_x, t_px_pos_y = random_position()


def update_generation(generation, players, children):
    # Calculate generation statistics
    active_players = [p for p in players if not (p.lose_status or p.win_status)]
    winners = [p for p in players if p.win_status]
    fitness_scores = [p.nn.fitness for p in players]
    
    avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
    max_fitness = max(fitness_scores) if fitness_scores else 0
    success_rate = len(winners) / len(players) if players else 0
    avg_steps = sum(p.steps for p in players) / len(players) if players else 0
    
    # Log the current generation with statistics
    print('Generation: ', generation)
    logger.info(f"Generation {generation}: "
                f"Avg Fitness={avg_fitness:.4f}, "
                f"Max Fitness={max_fitness:.4f}, "
                f"Success Rate={success_rate:.2%}, "
                f"Avg Steps={avg_steps:.1f}")
    
    # Print to console as well
    print(f"  Stats: Avg Fitness={avg_fitness:.4f}, Winners={len(winners)}/{len(players)}, Steps={avg_steps:.1f}")

    # Sort based on fitness
    fitness_ranking = sorted(players, key=lambda x: x.nn.fitness, reverse=True)

    # Elitism: preserve the top 5 performers
    elites = fitness_ranking[:5]
    for i in range(5):
        elites[i].nn.save(pathlib.Path(join(os.getcwd(), f'model_{i}.keras')))

    # Create new generation with tournament selection
    new_population = elites  # Start with the elites
    
    # Fill the rest of the population with children from tournament winners
    while len(new_population) < len(players):
        parent1 = tournament_selection(fitness_ranking)
        parent2 = tournament_selection(fitness_ranking)
        
        # Ensure parents are not the same for better diversity
        while parent1 == parent2:
            parent2 = tournament_selection(fitness_ranking)
            
        child_nn = gann.dynamic_crossover(parent1.nn, parent2.nn)
        gann.mutation(child_nn)  # Apply mutation to the child
        
        # Create a new player with the child's neural network
        child_player = Player()
        child_player.nn = child_nn
        new_population.append(child_player)

    # Replace old population with the new one
    for i in range(len(players)):
        players[i].nn = new_population[i].nn


def tournament_selection(population, tournament_size=3):
    """Select parent using tournament selection for better genetic diversity"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda p: p.nn.fitness)

def calculate_diversity_bonus(player, population):
    """Add small fitness bonus for unique behaviors to encourage diversity"""
    # Simple diversity measure: distance from average position
    avg_x = sum(p.x for p in population) / len(population)
    avg_y = sum(p.y for p in population) / len(population)
    
    position_uniqueness = math.sqrt((player.x - avg_x)**2 + (player.y - avg_y)**2)
    return min(0.1, position_uniqueness / 20.0)  # Cap at 10% bonus


def main():
    global step
    try:
        # Initialise screen
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('PYGAME AI')

        # Choose icons (convert to better performance)
        game_icon = pygame.image.load('deep-learning.png').convert_alpha()
        flag_icon = pygame.image.load('start_flag.png').convert_alpha()
        target_icon = pygame.image.load('target.png').convert_alpha()
        pygame.display.set_icon(game_icon)

        # Fill game background
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill(WHITE)

        # Font
        font = pygame.font.Font('./font/Roboto-Regular.ttf', 20)

        # Generate population
        players = []
        children = []

        # First time
        randomize_objectives()

        # The first distance is calculated by the Player's constructor
        for i in range(0, population):  # Create only AI players
            players.append(Player())
            children.append(Player())

        # Compile neural networks only once per architecture
        # Since all networks have the same architecture, compile just one as template
        template_nn = players[0].nn
        template_nn.compile()
        
        # Copy compilation state to all other networks
        for player in players[1:]:
            player.nn._is_compiled = True
            
        for child in children:
            child.nn._is_compiled = True

        # Generations loop
        for generation in range(0, generations):
            # Every generation clear screen and update
            screen.blit(background, (0, 0))
            pygame.display.flip()
            # Every generation, change the start and target's position to promote generalization
            randomize_objectives()
            # Reset all players positions after last play
            reset_players(players)

            # Create sprite group
            players_group = pygame.sprite.Group()
            for player in players:
                player.add(players_group)

            # Game loop
            render(screen, flag_icon, target_icon)
            update_text(screen, font, generation, players)
            running = True
            step = 0
            while running:
                players_group.draw(screen)
                update_text(screen, font, generation, players)
                pygame.display.flip()
                
                # Check if any AI player has won
                any_winner = any(player.win_status for player in players)
                if any_winner:
                    running = False
                
                # For user control - only ESC and QUIT events
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            running = False
                    # Check for QUIT event. If QUIT, then set running to false.
                    elif event.type == QUIT:
                        running = False

                # Update all AI players
                for p in range(0, len(players)):  # All players are AI now
                    if players[p].lose_status:
                        players[p].image.fill(PLAYER_BLUE)
                        continue
                    elif players[p].win_status:
                        players[p].image.fill(PLAYER_GREEN)
                        continue
                    move(players[p])
                    update_player(players[p], step)

                step += 1
                if step == 30:
                    running = False

            # Apply a penalty for hitting a wall
            for p in players:
                if p.lose_status:
                    p.nn.fitness *= 0.5  # Reduce fitness by 50% for failure

            update_generation(generation, players, children)

    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user. Exiting gracefully...")
        pygame.quit()
        print("Pygame closed successfully.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        pygame.quit()
        raise
    finally:
        # Ensure pygame is properly closed
        if pygame.get_init():
            pygame.quit()


if __name__ == '__main__':
    f_px_pos_x = 0
    f_px_pos_y = 0
    t_px_pos_x = 0
    t_px_pos_y = 0
    max_fitness = 0
    step = 0
    main()
