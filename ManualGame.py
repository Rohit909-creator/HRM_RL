import gym
import pygame
import numpy as np
import pickle
from datetime import datetime
import os

class ManualMountainCar:
    def __init__(self):
        self.env = gym.make('MountainCar-v0', render_mode='rgb_array')
        self.state = self.env.reset()[0]
        
        # Initialize Pygame
        pygame.init()
        self.screen_width, self.screen_height = 800, 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Manual Mountain Car Control')
        self.clock = pygame.time.Clock()
        
        # Storage for demonstrations
        self.demonstrations = []
        self.current_episode = []
        self.max_episodes = 20
        self.episode_count = 0
        
        # Font for rendering text
        self.font = pygame.font.SysFont(None, 24)
        
        # Create directory for saving demonstrations
        self.save_dir = "mountaincar_demos"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def render_screen(self):
        self.screen.fill((255, 255, 255))
        
        # Render the Mountain Car environment
        car_img = pygame.surfarray.make_surface(
            np.transpose(self.env.render(), (1, 0, 2))
        )
        car_img = pygame.transform.scale(car_img, (600, 400))
        self.screen.blit(car_img, (100, 50))
        
        # Render state information
        state_text = f"Car Position: {self.state[0]:.3f}, Car Velocity: {self.state[1]:.3f}"
        episode_text = f"Episode: {self.episode_count + 1}/{self.max_episodes}, Steps: {len(self.current_episode)}"
        
        # Add goal flag information
        goal_flag = "Goal Reached!" if self.state[0] >= 0.5 else "Goal: Reach position >= 0.5"
        
        text_surface = self.font.render(state_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (50, 470))
        
        text_surface = self.font.render(episode_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (50, 500))
        
        text_surface = self.font.render(goal_flag, True, (0, 128, 0) if self.state[0] >= 0.5 else (128, 0, 0))
        self.screen.blit(text_surface, (50, 530))
        
        # Instructions
        instructions = [
            "Left Arrow: Push Left (0)",
            "Right Arrow: Push Right (2)",
            "Down Arrow: No Push (1)",
            "R: Reset Environment",
            "S: Save Demonstrations",
            "Q: Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text_surface = self.font.render(instruction, True, (0, 0, 0))
            self.screen.blit(text_surface, (550, 470 + i * 25))
        
        pygame.display.flip()
    
    def reset_environment(self):
        self.state = self.env.reset()[0]
        if len(self.current_episode) > 0:
            self.demonstrations.append(self.current_episode)
            self.episode_count += 1
            print(f"Episode {self.episode_count} completed with {len(self.current_episode)} steps")
            self.current_episode = []
    
    def save_demonstrations(self):
        if len(self.demonstrations) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"mountaincar_demos_{timestamp}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(self.demonstrations, f)
            print(f"Saved {len(self.demonstrations)} demonstrations to {filename}")
    
    def run(self):
        running = True
        while running and self.episode_count < self.max_episodes:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_environment()
                    elif event.key == pygame.K_s:
                        self.save_demonstrations()
            
            # Get key states
            keys = pygame.key.get_pressed()
            
            # Default action is 1 (no push)
            action = 1
            
            if keys[pygame.K_LEFT]:
                action = 0  # Push car to the left
            elif keys[pygame.K_RIGHT]:
                action = 2  # Push car to the right
            elif keys[pygame.K_DOWN]:
                action = 1  # No push (neutral)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # Store the demonstration (state, action)
            self.current_episode.append((self.state, action, reward))
            
            # Update state
            self.state = next_state
            
            # Render
            self.render_screen()
            
            # Check if episode is done
            if terminated or truncated:
                self.reset_environment()
            
            self.clock.tick(30)  # Limit to 30 FPS
        
        # Save demonstrations if we have any
        if len(self.demonstrations) > 0 or len(self.current_episode) > 0:
            if len(self.current_episode) > 0:
                if reward:
                    self.demonstrations.append(self.current_episode)
            self.save_demonstrations()
        
        pygame.quit()

if __name__ == "__main__":
    manual_control = ManualMountainCar()
    manual_control.run()