from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
from setup import DEBUG_LEVEL

'''
Staff detection overview:
 - convert image to black&white
 - enhance image to help with blurry images
 - convert given image to numpy array for easier computation
 - iterate over array with 'probe window', that is smaller segment of array represents small vertical strip of the image
   - from probe window create histogram array using horizontal collapse
   - from histogram array filter out indexes of peaks
   - using obtained indexes of peaks in histograms find distance between staff lines
   - from obtained indexes and staff line distance get y coordinate of staves
 - from list of lists of y coordinates get single coordinate for every stave
 - return y coordinates of staves and stave line distance
'''

def print_position_list(list, x_coord):
  if len(list) == 0:
    print("(", len(list), ") ", x_coord, ": ---")
    return
  
  print("(", len(list), ") ", list[0][0], ":   ", end="")

  for i in list:
    print(i[1], ", ", end="")

  print()

def enhance_image(img):
  # simple image enhancer to help analize images
  img = cv2.GaussianBlur(img, (5, 5), 0)
  img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 55, 7)
  img = cv2.erode(img, (5, 5), 3)

  if DEBUG_LEVEL >=3:
    cv2.imshow("Debug image window", img)

  return img


def find_staff_line_distance_in_probe(peaks_list = []):
  space_between_peaks_list = []

  for i in range(1, len(peaks_list)):
    space_between_peaks_list.append(peaks_list[i] - peaks_list[i-1])

  if len(space_between_peaks_list) == 0:
    return 0

  avg = sum(space_between_peaks_list) // len(space_between_peaks_list)

  ret_list = []
  # filter out long sequences that can skew desired result
  for i in space_between_peaks_list:
    if i < avg:
      ret_list.append(i)    

  if len(ret_list) == 0:
    return 0

  return sum(ret_list) // len(ret_list)

# find index of peak in fragment of histogram array
def find_peak(hist_arr = np.array, idx_list = []):
  max_val_idx_list = []
  local_max = max(hist_arr[idx_list[0] : idx_list[-1] + 1])

  for i in idx_list:
    if hist_arr[i] == local_max:
      max_val_idx_list.append(i)

  staff_line_idx = sum(max_val_idx_list) // len(max_val_idx_list)

  return staff_line_idx

# find indexes of staff lines
def filter_out_stave_lines(max_val, hist_arr = np.array):
  cutoff = int(max_val * 0.65)
  peak_idx_list = []

  for i in range(0, hist_arr.size):
    if hist_arr[i] <= cutoff:
      hist_arr[i] = 0


  # iterate over histogram array, and collect indexes of peaks
  aux_list = []
  i = 0
  while i < hist_arr.size:
    if hist_arr[i] > cutoff:
      while i < hist_arr.size and hist_arr[i] > cutoff:
        aux_list.append(i)
        i += 1

      peak_idx_list.append(find_peak(hist_arr, aux_list))
      aux_list = []

    i += 1

  return peak_idx_list


def get_staves_positions(x_coord, distance, lines_idx_list = []) -> []:
  if len(lines_idx_list) < 5:
    return []

  staff_line_positions = []
  counter = 0
  aux_dist = lines_idx_list[1] - lines_idx_list[0]
  flag = aux_dist >= distance - 2 and aux_dist <= distance + 2

  # distance between lines is checked with some slack to help with non-uniform filtering 
  for i in range(1, len(lines_idx_list)):
    current_dist = lines_idx_list[i] - lines_idx_list[i-1]

    if flag and counter == 4:
      counter = 0
      flag = False
      staff_line_positions.append((x_coord, (lines_idx_list[i-1] + lines_idx_list[i-5]) // 2))
    elif current_dist >= distance - 2 and current_dist <= distance + 2 and not flag:
      counter = 1
      flag = True
    elif current_dist >= distance - 2 and current_dist <= distance + 2 and flag:
      counter += 1
    elif current_dist < distance - 2 or current_dist > distance + 2:
      flag = False
      counter = 0

  if flag and counter == 4:
    staff_line_positions.append((x_coord, (lines_idx_list[i] + lines_idx_list[i-4]) // 2))

  return staff_line_positions

def get_stave_y_coordinate(distance, positions_list = [[]]):
  y_coords = []

  distorted_dist = distance*5

  flat_list = sum(positions_list, [])
  flat_list.sort()

  aux_coord = flat_list[0]
  aux_list = []

  # in flattened list of positions group indexes that does not exceed daviation
  for i in flat_list:
    min_dev = aux_coord - distorted_dist
    max_dev = aux_coord + distorted_dist
    if i <= max_dev and i >= min_dev:
      aux_list.append(i)
    else:
      aux_coord = i
      if len(aux_list) == 0:
        continue

      y_coords.append(sum(aux_list) // len(aux_list))
      aux_list = []

  if len(aux_list) != 0:
    y_coords.append(sum(aux_list) // len(aux_list))

  return y_coords


def make_satves_points_list(in_list, distance):
  if len(in_list) == 0:
    return -1
  elif sum([len(i) for i in in_list]) == 0:
    return -1

  staves_points_list = []
  distorted_dist = distance*5

  flat_list = sum(in_list, [])
  flat_list.sort(key=lambda point: point[1])

  aux_coord = flat_list[0][1]
  aux_list = []
  min_dev = aux_coord - distorted_dist
  max_dev = aux_coord + distorted_dist

  for i in flat_list:
    if i[1] <= max_dev and i[1] >= min_dev:
      aux_list.append(i)
    else:
      aux_coord = i[1]
      min_dev = aux_coord - distorted_dist
      max_dev = aux_coord + distorted_dist

      if len(aux_list) < 3:
        aux_list = []
      else:
        staves_points_list.append(aux_list)
        aux_list = []

  for i in staves_points_list:
    i.sort(key=lambda point: point[0])

  return staves_points_list


# analizing seven slices of 1/30 of the image width to find staff line distance for further analisis of the image
def find_staff_line_distance(img_x, img_y, img_array = np.array):
  probe_window_size = img_x // 40
  probe_window_start_idx = probe_window_size * 5
  probe_window_hist_arr = np.zeros(img_y)
  staff_line_distance_list = []
  # iterate probe window over image, create histograms and get from them positions of staves
  for i in range(probe_window_start_idx, img_x, probe_window_size * 5):
    if i > img_x - probe_window_size * 2:
      break

    # count black pixels in probe window rows
    for j in range(i, i + probe_window_size):
      for k in range(0, img_y):
        if img_array[k][j] < 200:
          probe_window_hist_arr[k] += 1

    if DEBUG_LEVEL >= 3:
      plt.plot(probe_window_hist_arr)
      plt.title("Probe window histogram of y-pixels from ", i, " to ", i + probe_window_size)
      plt.ylabel("Number of dark pixels in row")
      plt.xlabel("Position on X axis of the image")
      plt.show()

    peaks_list = filter_out_stave_lines(probe_window_size, probe_window_hist_arr)

    # get distance between peaks
    staff_line_distance = find_staff_line_distance_in_probe(peaks_list)
    if staff_line_distance != 0:
      staff_line_distance_list.append(staff_line_distance)

  if len(staff_line_distance_list) != 0:
    return sum(staff_line_distance_list) // len(staff_line_distance_list)
  else:
    return -1
  

def find_staves_positions(img_x, img_y, staff_line_distance, stride = 1, img_array = np.array):
  probe_window_size = staff_line_distance * 2 + 5
  # offset for first probe window position to avoid unnecessary analizing of blank space at the edges of the sheet 
  probe_window_start_idx = probe_window_size * 2
  probe_window_hist_arr = np.zeros(img_y)
  staves_positions_list = []
  # iterate probe window over image, create histograms and get from them positions of staves
  for i in range(probe_window_start_idx, img_x, probe_window_size * stride):
    if i > img_x - probe_window_size * 2:
      break

    # count black pixels in probe window rows
    for j in range(i, i + probe_window_size):
      for k in range(0, img_y):
        if img_array[k][j] < 200:
          probe_window_hist_arr[k] += 1

    # filter obtained histogram data to single positions of lines
    peaks_list = filter_out_stave_lines(probe_window_size, probe_window_hist_arr)

    # with obtained data, find positions of staves in probe window and append to list of positions
    staves_positions = get_staves_positions(i, staff_line_distance, peaks_list)
    staves_positions_list.append(staves_positions)

    if DEBUG_LEVEL >= 1:
      print_position_list(staves_positions, i)

    if DEBUG_LEVEL >= 3:
      if len(staves_positions) < 8:
        plt.plot(probe_window_hist_arr)
        plt.title("Probe window histogram of y-pixels from ", i, " to ", i + probe_window_size)
        plt.ylabel("Number of dark pixels in row")
        plt.xlabel("Position on X axis of the image")
        plt.show()

    # reset probe window array
    probe_window_hist_arr = np.zeros(img_y)

  if len(staves_positions_list) == 0:
    return -1
  else:
    return staves_positions_list
  

def find_staves(img, stride = 2):
  img = enhance_image(img)
  print("-- Image enhancement done --")

  # convert image to array to help in image processing in search of staves
  img_array = tf.keras.utils.img_to_array(img)
  img_y, img_x, c = img_array.shape
  print("-- Conversion of image to array done --")

  staff_line_dist = find_staff_line_distance(img_x, img_y, img_array)
  if staff_line_dist == -1:
    return -1, 0
  print("-- Finding staff line distance done --")
  print("-- Staff line distance: ", staff_line_dist)

  staves_positions = find_staves_positions(img_x, img_y, staff_line_dist, stride, img_array)
  print(staves_positions)
  if len(staves_positions) == 0:
    return -1, -1
  else:
    print("-- Finding positions of staves done --")
    staves_points_list = make_satves_points_list(staves_positions, staff_line_dist)
    if len(staves_points_list) == 0:
      return -1, -1
    print("-- Reformatting of staves position list done --")
    return staves_points_list, staff_line_dist


