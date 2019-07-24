import os
import shutil
import json
import uuid
import pyautogui as pag
import numpy as np
import pyscreenshot as ImageGrab


def empty_dir(dir_path):
  for fname in os.listdir(dir_path):
    fpath = os.path.join(dir_path, fname)
    if os.path.isfile(fpath):
      os.unlink(fpath)
    elif os.path.isdir(fpath):
      shutil.rmtree(fpath)


def generate_uuid():
  return str(uuid.uuid4())


def read_json(fpath):
  with open(fpath, 'r') as f:
    return json.load(f)


def read_grab_area():
  d = read_json('grab_area.json')
  return (*d['topLeft'], *d['bottomRight'])


def get_hw(grab_area):
  w = grab_area[2] - grab_area[0]
  h = grab_area[3] - grab_area[1]
  return h, w


def refresh(grab_area):
  drag_x = (grab_area[0] + grab_area[2]) // 2
  margin = 30
  drag_y_start = grab_area[1] + margin
  drag_y_end = grab_area[3] + margin
  pag.moveTo(drag_x, drag_y_start)
  pag.dragTo(drag_x, drag_y_end, duration=0.2, button='left')


def grab_screen(grab_area=None):
  img = ImageGrab.grab(bbox=grab_area)
  img = np.asarray(img)[:, :, ::-1]  # RGB to BGR
  return img
