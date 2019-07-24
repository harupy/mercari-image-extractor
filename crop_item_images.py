import os
import time
import cv2

from tqdm import tqdm
from config import IMAGE_DIR
from utils import (mkdir_if_not_exists,
                   empty_dir,
                   generate_uuid,
                   read_grab_area,
                   refresh,
                   grab_screen
                   )

GREEN = (0, 255, 0)


def find_contours(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray, 200, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  return contours


def filter_contours(contours):
  return list(filter(lambda cnt: 1000 < cv2.contourArea(cnt) < 50000, contours))


def draw_contours(img, contours, hierarchy=None):
  img_cnt = img.copy()
  for cnt in contours:
    cv2.drawContours(img_cnt, [cnt], 0, GREEN, 1, cv2.LINE_8, hierarchy, 0)
  return img_cnt


def draw_crop_areas(img, crop_areas):
  img_ca = img.copy()
  for x, y, w, h in crop_areas:
    cv2.rectangle(img_ca, (x, y), (x + w, y + h), GREEN, 2)

  cv2.imwrite('crop_areas.png', img_ca)


def crop_images(img, crop_areas):
  pad = 3
  imgs = []
  for (x, y, w, h) in crop_areas:
    portion = img[y + pad: y + h - pad, x + pad: x + w - pad, :]
    imgs.append(portion)
  return imgs


def find_crop_areas(grid_img_path='grid.png'):
  grid = cv2.imread(grid_img_path)
  grid = cv2.imread('grid.png')
  contours = find_contours(grid)
  contours = filter_contours(contours)
  return list(map(cv2.boundingRect, contours))


def main():
  mkdir_if_not_exists(IMAGE_DIR)
  empty_dir(IMAGE_DIR)
  grab_area = read_grab_area()
  crop_areas = find_crop_areas()
  for _ in tqdm(range(2 ** 11)):
    ss = grab_screen(grab_area)
    draw_crop_areas(ss, crop_areas)
    for img in crop_images(ss, crop_areas):
      save_path = os.path.join(IMAGE_DIR, generate_uuid() + '.png')
      cv2.imwrite(save_path, img)
    refresh(grab_area)
    time.sleep(1)


if __name__ == '__main__':
  main()
