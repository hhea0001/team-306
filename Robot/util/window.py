
import os
import numpy as np
import pygame
import pygame.display as display

TEXT_LEFT = 0
TEXT_RIGHT = 1
TEXT_CENTRE = 2

class Window:
    def __init__(self):
        self.quit = False
        # Setup canvas
        self.width, self.height = 700, 660
        self.canvas = display.set_mode((self.width, self.height))
        self.canvas.fill((0,0,0))
        display.set_caption("Team 306 Robot")
        display.update()
        # Setup font
        dirname = os.path.relpath(os.path.dirname(__file__), start = os.curdir)
        font_file = dirname + '/fonts/Roboto-Regular.ttf'
        pygame.font.init()
        self.font = pygame.font.Font(font_file, 24)

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