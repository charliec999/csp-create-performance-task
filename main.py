import pygame
import numpy as np
import numba
from numba import njit

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ecosystem Simulation")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)

@njit
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def move_towards(x, y, tx, ty, speed):
    dx, dy = tx - x, ty - y
    dist = np.sqrt(dx**2 + dy**2)
    if dist > 0:
        x += dx/dist * speed
        y += dy/dist * speed
    return x, y

class Ecosystem:
    def __init__(self):
        self.plant_positions = np.random.rand(300, 2) * np.array([WIDTH, HEIGHT])
        self.prey_positions = np.random.rand(100, 2) * np.array([WIDTH, HEIGHT])
        self.predator_positions = np.random.rand(25, 2) * np.array([WIDTH, HEIGHT])
        
        self.prey_energies = np.ones(100) * 100
        self.predator_energies = np.ones(25) * 150
        
        self.plant_colors = np.array([50, 200, 50])
        self.prey_colors = np.array([50, 50, 200])
        self.predator_colors = np.array([200, 50, 50])

    def update(self):
        self.constrain_positions()
        self.update_plants()
        self.update_prey()
        self.update_predators()
        self.handle_repopulation()

    def constrain_positions(self):
        for positions in [self.plant_positions, self.prey_positions, self.predator_positions]:
            positions[:, 0] = np.clip(positions[:, 0], 0, WIDTH)
            positions[:, 1] = np.clip(positions[:, 1], 0, HEIGHT - 50)

    def update_plants(self):
        if len(self.plant_positions) < 500 and np.random.random() < 0.1:
            new_plant = np.random.rand(1, 2) * np.array([WIDTH, HEIGHT - 50])
            self.plant_positions = np.vstack((self.plant_positions, new_plant))

    def update_prey(self):
        to_remove = []
        to_reproduce = []
        for i in range(len(self.prey_positions)):
            nearest_predator_dist = np.inf
            nearest_plant_dist = np.inf
            nearest_plant_index = -1
            nearest_predator_index = -1

            for j, predator_pos in enumerate(self.predator_positions):
                dist = distance(self.prey_positions[i][0], self.prey_positions[i][1], 
                                predator_pos[0], predator_pos[1])
                if dist < nearest_predator_dist:
                    nearest_predator_dist = dist
                    nearest_predator_index = j

            for j, plant_pos in enumerate(self.plant_positions):
                dist = distance(self.prey_positions[i][0], self.prey_positions[i][1], 
                                plant_pos[0], plant_pos[1])
                if dist < nearest_plant_dist:
                    nearest_plant_dist = dist
                    nearest_plant_index = j

            if nearest_predator_dist < 150:
                escape_x = 2 * self.prey_positions[i][0] - self.predator_positions[nearest_predator_index][0]
                escape_y = 2 * self.prey_positions[i][1] - self.predator_positions[nearest_predator_index][1]
                self.prey_positions[i] = move_towards(
                    self.prey_positions[i][0], self.prey_positions[i][1],
                    escape_x, escape_y, 6
                )
            elif nearest_plant_index != -1 and nearest_plant_dist < 50:
                self.prey_positions[i] = move_towards(
                    self.prey_positions[i][0], self.prey_positions[i][1],
                    self.plant_positions[nearest_plant_index][0],
                    self.plant_positions[nearest_plant_index][1],
                    4
                )
                if nearest_plant_dist < 10:
                    self.plant_positions = np.delete(self.plant_positions, nearest_plant_index, axis=0)
                    self.prey_energies[i] += 30
                    to_reproduce.append(i)

            self.prey_energies[i] -= 0.5
            if self.prey_energies[i] <= 0:
                to_remove.append(i)

        for idx in to_reproduce:
            if np.random.random() < 0.3:
                new_prey = self.prey_positions[idx] + np.random.normal(0, 10, 2)
                new_prey[0] = max(0, min(new_prey[0], WIDTH))
                new_prey[1] = max(0, min(new_prey[1], HEIGHT - 50))
                self.prey_positions = np.vstack((self.prey_positions, new_prey))
                self.prey_energies = np.append(self.prey_energies, 100)

        self.prey_positions = np.delete(self.prey_positions, to_remove, axis=0)
        self.prey_energies = np.delete(self.prey_energies, to_remove)

    def update_predators(self):
        to_remove = []
        to_reproduce = []
        for i in range(len(self.predator_positions)):
            nearest_prey_dist = np.inf
            nearest_prey_index = -1

            for j, prey_pos in enumerate(self.prey_positions):
                dist = distance(self.predator_positions[i][0], self.predator_positions[i][1], 
                                prey_pos[0], prey_pos[1])
                if dist < nearest_prey_dist:
                    nearest_prey_dist = dist
                    nearest_prey_index = j

            if nearest_prey_index != -1:
                self.predator_positions[i] = move_towards(
                    self.predator_positions[i][0], self.predator_positions[i][1],
                    self.prey_positions[nearest_prey_index][0],
                    self.prey_positions[nearest_prey_index][1],
                    2
                )
                if nearest_prey_dist < 20:
                    self.prey_positions = np.delete(self.prey_positions, nearest_prey_index, axis=0)
                    self.predator_energies[i] += 50
                    to_reproduce.append(i)

            self.predator_energies[i] -= 1
            if self.predator_energies[i] <= 0:
                to_remove.append(i)

        for idx in to_reproduce:
            if np.random.random() < 0.2:
                new_predator = self.predator_positions[idx] + np.random.normal(0, 10, 2)
                new_predator[0] = max(0, min(new_predator[0], WIDTH))
                new_predator[1] = max(0, min(new_predator[1], HEIGHT - 50))
                self.predator_positions = np.vstack((self.predator_positions, new_predator))
                self.predator_energies = np.append(self.predator_energies, 150)

        self.predator_positions = np.delete(self.predator_positions, to_remove, axis=0)
        self.predator_energies = np.delete(self.predator_energies, to_remove)

    def handle_repopulation(self):
        pass  # Removed as reproduction is now handled in update methods

    def draw(self, surface):
        surface.fill((255, 255, 255))
        
        for pos in self.plant_positions:
            pygame.draw.circle(surface, self.plant_colors, (int(pos[0]), int(pos[1])), 3)
        
        for pos in self.prey_positions:
            pygame.draw.circle(surface, self.prey_colors, (int(pos[0]), int(pos[1])), 5)
        
        for pos in self.predator_positions:
            pygame.draw.circle(surface, self.predator_colors, (int(pos[0]), int(pos[1])), 8)

def draw_button(surface, text, x, y, width=100, height=50):
    button_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, (200, 200, 200), button_rect)
    pygame.draw.rect(surface, (100, 100, 100), button_rect, 2)
    text_surface = font.render(text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=button_rect.center)
    surface.blit(text_surface, text_rect)
    return {"rect": button_rect, "text": text}

def main():
    ecosystem = Ecosystem()
    running = True
    
    buttons = [
        draw_button(screen, "+Prey", 10, HEIGHT - 50),
        draw_button(screen, "-Prey", 120, HEIGHT - 50),
        draw_button(screen, "+Pred", 230, HEIGHT - 50),
        draw_button(screen, "-Pred", 340, HEIGHT - 50),
        draw_button(screen, "+Plant", 450, HEIGHT - 50),
        draw_button(screen, "-Plant", 560, HEIGHT - 50),
        draw_button(screen, "Reset", WIDTH - 110, HEIGHT - 50)
    ]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    if button["rect"].collidepoint(mouse_pos):
                        if button["text"] == "+Prey":
                            for _ in range(10):
                                new_prey = np.random.rand(1, 2) * np.array([WIDTH, HEIGHT - 50])
                                ecosystem.prey_positions = np.vstack((ecosystem.prey_positions, new_prey))
                                ecosystem.prey_energies = np.append(ecosystem.prey_energies, 100)
                        elif button["text"] == "-Prey" and len(ecosystem.prey_positions) > 10:
                            ecosystem.prey_positions = ecosystem.prey_positions[:-10]
                            ecosystem.prey_energies = ecosystem.prey_energies[:-10]
                        elif button["text"] == "+Pred":
                            for _ in range(10):
                                new_predator = np.random.rand(1, 2) * np.array([WIDTH, HEIGHT - 50])
                                ecosystem.predator_positions = np.vstack((ecosystem.predator_positions, new_predator))
                                ecosystem.predator_energies = np.append(ecosystem.predator_energies, 150)
                        elif button["text"] == "-Pred" and len(ecosystem.predator_positions) > 5:
                            ecosystem.predator_positions = ecosystem.predator_positions[:-10]
                            ecosystem.predator_energies = ecosystem.predator_energies[:-10]
                        elif button["text"] == "+Plant":
                            for _ in range(10):
                                new_plant = np.random.rand(1, 2) * np.array([WIDTH, HEIGHT - 50])
                                ecosystem.plant_positions = np.vstack((ecosystem.plant_positions, new_plant))
                        elif button["text"] == "-Plant" and len(ecosystem.plant_positions) > 10:
                            ecosystem.plant_positions = ecosystem.plant_positions[:-10]
                        elif button["text"] == "Reset":
                            ecosystem = Ecosystem()

        ecosystem.update()
        ecosystem.draw(screen)
        
        for button in buttons:
            draw_button(screen, button["text"], button["rect"].x, button["rect"].y)
            
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main()
