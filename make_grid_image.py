
import time
from tqdm import tqdm
import numpy as np
import cv2
from utils import read_grab_area, get_hw, refresh, grab_screen


def make_grid_image(grab_area):
  img = np.zeros(get_hw(grab_area))
  for i in tqdm(range(50)):
    refresh(grab_area)
    time.sleep(0.5)
    ss = grab_screen(grab_area)
    gray = cv2.cvtColor(ss, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, 0)
    img += 255 - thresh
    img = np.clip(img, 0, 255)

  cv2.imwrite('grid.png', 255 - img)


def main():
  grab_area = read_grab_area()
  make_grid_image(grab_area)


if __name__ == '__main__':
  main()
