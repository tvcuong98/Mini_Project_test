import pyautogui
import time

def click_screen(x, y):
    pyautogui.click(x, y)

# Coordinates of the point you want to click
click_x = 500
click_y = 700

# Delay between each click in seconds
delay = 120

# Infinite loop to continuously click every 2 minutes
while True:
    click_screen(click_x, click_y)
    time.sleep(delay)
