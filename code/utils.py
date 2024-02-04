import cv2
import os

def print_list(list):
  for i in list:
    print(i)

def debug_show(name, step, text, display):
    path = 'D:\Kody\Pracka_inzynierska\img_dump'
    filetext = text.replace(' ', '_')
    outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
    cv2.imwrite(os.path.join(path, outfile), display)