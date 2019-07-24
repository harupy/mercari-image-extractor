from glob import glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from config import IMAGE_DIR, TENSORBORAD_DIR


def main():
  img_paths = glob(f'{IMAGE_DIR}/dummy/*.png')
  nrows = 32
  ncols = 32
  num_files = nrows * ncols

  tile_width = 50
  tile_height = 50

  tiles = np.zeros((nrows * tile_height, ncols * tile_width, 3))

  for idx, img_path in enumerate(tqdm(sorted(img_paths[:num_files]))):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (tile_width, tile_height))
    idx_row = idx // ncols
    idx_col = idx - (idx_row * ncols)

    row_start = idx_row * tile_height
    row_end = (idx_row + 1) * tile_height
    col_start = idx_col * tile_width
    col_end = (idx_col + 1) * tile_width

    tiles[row_start: row_end, col_start: col_end] = img

  cv2.imwrite(f'{TENSORBORAD_DIR}/tiles.png', tiles)

  df = pd.read_csv(f'{TENSORBORAD_DIR}/vectors_all.tsv')
  df.iloc[:num_files].to_csv(f'{TENSORBORAD_DIR}/vectors.tsv', index=False, header=False)


if __name__ == '__main__':
  main()
