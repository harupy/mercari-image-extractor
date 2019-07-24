import time
import pyautogui as pag


def main():
  while True:
    print(pag.position())
    time.sleep(1)


if __name__ == '__main__':
  main()
