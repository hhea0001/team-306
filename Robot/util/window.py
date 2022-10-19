
import os
import numpy as np
import pygame
import pygame.display as display
from util.planning import Planner
from util.sim import SimRobot, Simulation

TEXT_LEFT = 0
TEXT_RIGHT = 1
TEXT_CENTRE = 2

class Window:
    def __init__(self):
        self.quit = False
        # Setup canvas
        self.width, self.height = 800, 524
        self.canvas = display.set_mode((self.width, self.height))
        self.canvas.fill((0,0,0))
        display.set_caption("Team 306 Robot")
        display.update()
        # Setup font
        dirname = os.path.relpath(os.path.dirname(__file__), start = os.curdir)
        font_file = dirname + '/fonts/Roboto-Regular.ttf'
        pygame.font.init()
        self.font = pygame.font.Font(font_file, 24)
        # Setup background
        background_file = dirname + '/images/background.png'
        self.background = pygame.image.load(background_file)
        self.draw_background()
        robot_file = dirname + '/images/robot.png'
        self.robot = pygame.image.load(robot_file)
        self.robot = pygame.transform.smoothscale(self.robot, (30, 30))
    
    def __world_to_map(self, coord):
        return [-coord[0] * 258/1.5 + 538, coord[1] * 260/1.5 + 262]

    def update(self):
        # Update frame
        display.update()
        # Clear and resize
        self.canvas.fill((0,0,0))
        width, height = display.get_window_size()
        if width != self.width or height != self.height:
            self.canvas = display.set_mode((self.width, self.height))
        # Quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                display.quit()
                pygame.quit()
                self.quit = True
    
    def draw_text(self, string, location, colour = (255,255,255), centre: TEXT_LEFT | TEXT_RIGHT | TEXT_CENTRE = TEXT_LEFT):
        text = self.font.render(string, True, colour)
        x, y = location
        if centre == TEXT_RIGHT:
            x -= text.get_width()
        elif centre == TEXT_CENTRE:
            x -= text.get_width() / 2
        self.canvas.blit(text, (x, y))

    def draw_image(self, image: np.ndarray):
        cv2_img = np.rot90(image)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.scale(view, (320, 240))
        view = pygame.transform.flip(view, True, False)
        self.canvas.blit(view, (0,0))
    
    def draw_background(self):
        self.canvas.blit(self.background, (0, 0))
    
    def draw_robot(self, robot: SimRobot):
        robot_pic = pygame.transform.rotate(self.robot, robot.get_angle() * 180 / np.pi)
        robot_pos = robot.get_position()
        robot_pos = self.__world_to_map([robot_pos[0], robot_pos[1]])
        robot_pos[0] -= robot_pic.get_width() / 2 
        robot_pos[1] -= robot_pic.get_height() / 2
        self.canvas.blit(robot_pic, robot_pos)

    def draw_landmarks(self, taglist, landmarks):
        for key in taglist:
            i = taglist[key]
            position = [landmarks[0,i], landmarks[1,i]]
            if "apple" in key:
                colour = (255, 0, 0)
                size = 7
            if "lemon" in key:
                colour = (255, 255, 0)
                size = 7
            if "orange" in key:
                colour = (255, 128, 0)
                size = 7
            if "pear" in key:
                colour = (0, 255, 0)
                size = 7
            if "strawberry" in key:
                colour = (255, 0, 0)
                size = 4
            if "aruco" in key:
                colour = (255, 255, 255)
                size = 7
            pygame.draw.circle(self.canvas, colour, self.__world_to_map(position), size, 2)

    def draw_plan(self, plan):
        pygame.draw.lines(self.canvas, (255, 255, 255), False, [self.__world_to_map(x) for x in plan], width=2)

    def draw_camera(self, marked_image):
        cv2_img = np.rot90(marked_image)
        view = pygame.surfarray.make_surface(cv2_img)
        #view = pygame.transform.scale(view, (320, 240))
        view = pygame.transform.smoothscale(view, (272, 204))
        view = pygame.transform.flip(view, True, False)
        self.canvas.blit(view, (4,316))

    def draw(self, simulation: Simulation, marked_image: np.ndarray, plan: Planner):
        self.draw_background()
        self.draw_camera(marked_image)
        self.draw_landmarks(simulation.taglist, simulation.landmarks)
        self.draw_robot(simulation.robot)
        if len(plan.current_plan) > 0:
            self.draw_plan(plan.current_plan)
        pos = simulation.get_position()
        angle = simulation.get_angle()
        # Draw current state
        self.draw_text(f"x: {pos[0]:2.2f}", (30, 110))
        self.draw_text(f"y: {pos[1]:2.2f}", (30, 140))
        self.draw_text(f"θ: {angle:2.2f}", (30, 170))