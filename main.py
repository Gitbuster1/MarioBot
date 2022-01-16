import time

import cv2
import numpy as np
import mss
from matplotlib import pyplot as plt
from pynput import keyboard, mouse
import keyboard as kb

from Model.DQNAgent import DQNAgent


def find_object(game_area, obj_img, threshold):
    result = cv2.matchTemplate(game_area, obj_img, cv2.TM_CCOEFF_NORMED)
    width = obj_img.shape[1]
    height = obj_img.shape[0]
    y_loc, x_loc = np.where(result >= threshold)
    rectangles = []
    for (x, y) in zip(x_loc, y_loc):
        rectangles.append([int(x), int(y), int(width), int(height)])
        rectangles.append([int(x), int(y), int(width), int(height)])
    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)
    return rectangles


def check_for_enemy(game_area, enemy1_img, threshold):
    rectangles = find_object(game_area, enemy1_img, threshold)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(game_area, (x, y), (x + w, y + h), (255, 0, 0), 2)


def check_for_flag(game_area, flag_img, threshold):
    rectangles = find_object(game_area, flag_img, threshold)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(game_area, (x, y), (x + w, y + h), (0, 255, 0), 2)


def check_for_castle(game_area, castle_img, threshold):
    rectangles = find_object(game_area, castle_img, threshold)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(game_area, (x, y), (x + w, y + h), (0, 0, 255), 2)


def capture_score(game_area, score_title_img, threshold):
    rectangles = find_object(game_area, score_title_img, threshold)
    no_digits = 6
    for (x, y, w, h) in rectangles:
        w_digit = int(w / no_digits)
        h_digit = h
        x_digit = x
        y_digit = y + h
        cv2.rectangle(game_area, (x_digit, y_digit), (x_digit + w_digit * no_digits, y_digit + h_digit),
                      (0, 0, 0), 2)
        digits = []
        for i in range(no_digits):
            digits.append((x_digit + i * w_digit, y_digit))
        recognize_digit(digits, w_digit, h_digit)


def capture_time(game_area, time_title_img, threshold):
    rectangles = find_object(game_area, time_title_img, threshold)
    no_digits = 3
    for (x, y, w, h) in rectangles:
        w_digit = int(w / (no_digits + 1))
        h_digit = h
        x_digit = x + w_digit
        y_digit = y + h
        cv2.rectangle(game_area, (x_digit, y_digit), (x_digit + w_digit * no_digits, y_digit + h_digit),
                      (0, 0, 0), 2)
        digits = []
        for i in range(no_digits):
            digits.append((x_digit + i * w_digit, y_digit))
        recognize_digit(digits, w_digit, h_digit)


def recognize_digit(digits, w_digit, h_digit):
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 1, 0): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    for idx in range(len(digits)):
        x, y = digits[idx]
        segments = [
            ((x, y), (x + w_digit, y + int(h_digit * 0.2))),  # top
            ((x, y), (x + int(w_digit * 0.25), y + int(h_digit / 2))),  # top-left
            ((x + int(w_digit * 0.75), y), (x + w_digit, y + int(h_digit / 2))),  # top-right
            ((x, y + int(h_digit / 2 * 0.9)), (x + w_digit, y + int(h_digit / 2 * 1.1))),  # center
            ((x, y + int(h_digit / 2)), (x + int(w_digit * 0.25), y + h_digit)),  # bottom-left
            ((x + int(w_digit * 0.75), y + int(h_digit / 2)), (x + w_digit, y + h_digit)),  # bottom-right
            ((x, int((y + h_digit) * 0.8)), (x + w_digit, y + h_digit))  # bottom
        ]
        on = [0] * len(segments)

        # # loop over the segments
        # for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        #     # extract the segment ROI, count the total number of
        #     # thresholded pixels in the segment, and then compute
        #     # the area of the segment
        #     segROI = roi[yA:yB, xA:xB]
        #     total = cv2.countNonZero(segROI)
        #     area = (xB - xA) * (yB - yA)
        #     # if the total number of non-zero pixels is greater than
        #     # 50% of the area, mark the segment as "on"
        #     if total / float(area) > 0.5:
        #         on[i] = 1
        # # lookup the digit and draw it on the image
        # digit = DIGITS_LOOKUP[tuple(on)]
        # digits.append(digit)
        # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.putText(output, str(digit), (x - 10, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def build_number():
    pass


def close_app():
    exit()


# def train_agent():
#     num_episodes = 2000
#     agent = Agent()
#     env = Grid()
#     rewards = []
#     for _ in range(num_episodes):
#         state = env.reset()
#         episode_reward = 0
#         while True:
#             action_id, action = agent.act(state)
#             next_state, reward, terminal = env.step(action)
#             episode_reward += reward
#
#             agent.q_update(state, action_id, reward, next_state, terminal)
#             state = next_state
#
#             if terminal:
#                 break
#         rewards.append(episode_reward)
#
#     plt.plot(rewards)
#     plt.show()
#     return agent.best_policy()


def run():
    key = keyboard.Controller()
    enemy1_img = cv2.imread("Resources/enemy1.png", cv2.IMREAD_GRAYSCALE)
    # enemy1_img = cv2.GaussianBlur(enemy1_img, (5, 5), 0)
    # enemy1_img = cv2.Canny(enemy1_img, 50, 200, 255)
    # enemy1_img = cv2.cvtColor(enemy1_img, cv2.COLOR_BGR2GRAY)
    enemy2_img = cv2.imread("Resources/enemy2.png", cv2.IMREAD_GRAYSCALE)
    enemy2_flipped_img = cv2.imread("Resources/enemy2_flipped.png", cv2.IMREAD_GRAYSCALE)
    # enemy2_img = cv2.cvtColor(enemy2_img, cv2.COLOR_BGR2GRAY)
    flag_img = cv2.imread("Resources/flag.png", cv2.IMREAD_GRAYSCALE)
    # flag_img = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY)
    castle_img = cv2.imread("Resources/castle.png", cv2.IMREAD_GRAYSCALE)
    # castle_img = cv2.cvtColor(castle_img, cv2.COLOR_BGR2GRAY)
    score_title_img = cv2.imread("Resources/score_title.png", cv2.IMREAD_GRAYSCALE)
    time_title_img = cv2.imread("Resources/time_title.png", cv2.IMREAD_GRAYSCALE)
    # TODO: change this to a dynamic version, not static (so it can work on different devices and screens)
    with mss.mss() as mss_instance:
        monitor = mss_instance.monitors[2]
        monitor.__init__({'top': 31, 'left': 3168, 'width': 672, 'height': 672})
    while True:
        screenshot = mss_instance.grab(monitor)
        game_area = np.array(screenshot)
        game_area = cv2.cvtColor(game_area, cv2.COLOR_BGR2GRAY)
        # game_area = cv2.GaussianBlur(game_area, (5, 5), 0)
        # game_area = cv2.Canny(game_area, 50, 200, 255)

        check_for_enemy(game_area, enemy1_img, threshold=0.46)
        check_for_enemy(game_area, enemy2_img, threshold=0.46)
        check_for_enemy(game_area, enemy2_flipped_img, threshold=0.53)
        check_for_flag(game_area, flag_img, threshold=0.55)
        capture_score(game_area, score_title_img, threshold=0.9)
        capture_time(game_area, time_title_img, threshold=0.9)
        # check_for_castle(game_area, castle_img, threshold=0.5)

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

# ----------
# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = wrapper(env)

# Parameters
states = (84, 84, 4)
actions = env.action_space.n

# Agent
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)

# Episodes
episodes = 10000
rewards = []

# Timing
start = time.time()
step = 0

# Main loop
for e in range(episodes):

    # Reset env
    state = env.reset()

    # Reward
    total_reward = 0
    iter = 0

    # Play
    while True:

        # Show env (disabled)
        # env.render()

        # Run agent
        action = agent.run(state=state)

        # Perform action
        next_state, reward, done, info = env.step(action=action)

        # Remember transition
        agent.add(experience=(state, next_state, action, reward, done))

        # Update agent
        agent.learn()

        # Total reward
        total_reward += reward

        # Update state
        state = next_state

        # Increment
        iter += 1

        # If done break loop
        if done or info['flag_get']:
            break

    # Rewards
    rewards.append(total_reward / iter)

    # Print
    if e % 100 == 0:
        print('Episode {e} - +'
              'Frame {f} - +'
              'Frames/sec {fs} - +'
              'Epsilon {eps} - +'
              'Mean Reward {r}'.format(e=e,
                                       f=agent.step,
                                       fs=np.round((agent.step - step) / (time.time() - start)),
                                       eps=np.round(agent.eps, 4),
                                       r=np.mean(rewards[-100:])))
        start = time.time()
        step = agent.step

# Save rewards
np.save('rewards.npy', rewards)
