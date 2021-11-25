import time
import cv2
import numpy as np
import mss
from pynput import keyboard, mouse
import keyboard as kb


def find_object(game_area, obj_img):
    result = cv2.matchTemplate(game_area, obj_img, cv2.TM_CCOEFF_NORMED)
    width = obj_img.shape[1]
    height = obj_img.shape[0]
    # TODO: to be tested more thoroughly to get the best value for the threshold
    threshold = .55
    y_loc, x_loc = np.where(result >= threshold)
    rectangles = []
    for (x, y) in zip(x_loc, y_loc):
        rectangles.append([int(x), int(y), int(width), int(height)])
        rectangles.append([int(x), int(y), int(width), int(height)])
    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)
    return rectangles


def check_for_enemy(game_area, enemy_img):
    rectangles = find_object(game_area, enemy_img)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(game_area, (x, y), (x + w, y + h), (255, 0, 0), 2)


def check_for_flag(game_area, flag_img):
    rectangles = find_object(game_area, flag_img)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(game_area, (x, y), (x + w, y + h), (0, 255, 0), 2)


def check_for_castle(game_area, castle_img):
    rectangles = find_object(game_area, castle_img)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(game_area, (x, y), (x + w, y + h), (0, 0, 255), 2)


def close_app():
    exit()


def run():
    key = keyboard.Controller()
    enemy1_img = cv2.imread("enemy1.png", cv2.IMREAD_UNCHANGED)
    enemy1_img = cv2.cvtColor(enemy1_img, cv2.COLOR_BGR2GRAY)
    enemy2_img = cv2.imread("enemy2.png", cv2.IMREAD_UNCHANGED)
    enemy2_img = cv2.cvtColor(enemy2_img, cv2.COLOR_BGR2GRAY)
    flag_img = cv2.imread("flag.png", cv2.IMREAD_UNCHANGED)
    flag_img = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY)
    castle_img = cv2.imread("castle.png", cv2.IMREAD_UNCHANGED)
    castle_img = cv2.cvtColor(castle_img, cv2.COLOR_BGR2GRAY)
    # TODO: change this to a dynamic version, not static (so it can work on multiple devices and screens)
    with mss.mss() as mss_instance:
        monitor = mss_instance.monitors[2]
        monitor.__init__({'top': 31, 'left': 3168, 'width': 672, 'height': 672})
    while True:
        screenshot = mss_instance.grab(monitor)
        game_area = np.array(screenshot)
        game_area = cv2.cvtColor(game_area, cv2.COLOR_BGR2GRAY)

        check_for_enemy(game_area, enemy1_img)
        check_for_enemy(game_area, enemy2_img)
        check_for_flag(game_area, flag_img)
        check_for_castle(game_area, castle_img)

        cv2.imshow('Screen', game_area)
        cv2.waitKey(1)
        if kb.is_pressed("x"):
            close_app()
        if kb.is_pressed("f"):
            pass
        if kb.is_pressed("F10"):
            pass
        if kb.is_pressed("left arrow"):
            pass
        if kb.is_pressed("right arrow"):
            pass


run()
