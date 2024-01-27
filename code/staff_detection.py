from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


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

def enhance_image(img = Image):
  # simple image enhancer to help analize blurry images
  contrast_factor = 1.6
  contrast_enhancer = ImageEnhance.Contrast(img)
  img = contrast_enhancer.enhance(contrast_factor)

  sharpness_factor = 3
  sharpness_enhancer = ImageEnhance.Sharpness(img)
  img = sharpness_enhancer.enhance(sharpness_factor)

  return img


def find_staff_line_distance(peaks_list = []):
  space_between_peaks_list = []

  for i in range(1, len(peaks_list)):
    space_between_peaks_list.append(peaks_list[i] - peaks_list[i-1])

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
  cutoff = int(max_val * 0.9)
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


def get_staves_positions(distance, lines_idx_list = []) -> []:
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
      staff_line_positions.append((lines_idx_list[i-1] + lines_idx_list[i-5]) // 2)
    elif current_dist >= distance - 2 and current_dist <= distance + 2 and not flag:
      counter = 1
      flag = True
    elif current_dist >= distance - 2 and current_dist <= distance + 2 and flag:
      counter += 1
    elif current_dist < distance - 2 or current_dist > distance + 2:
      flag = False
      counter = 0

  if flag and counter == 4:
    counter = 0
    flag = False
    staff_line_positions.append((lines_idx_list[i] + lines_idx_list[i-4]) // 2)

  return staff_line_positions

def get_stave_y_coordinate(distance, positions_list = [[]]):
  y_coords = []

  flat_list = sum(positions_list, [])
  flat_list.sort()

  aux_coord = flat_list[0]
  aux_list = []

  # in flattened list of positions group indexes that does not exceed 
  for i in flat_list:
    min_dev = aux_coord - distance
    max_dev = aux_coord + distance
    if i <= max_dev and i >= min_dev:
      aux_list.append(i)
    else:
      aux_coord = i
      y_coords.append(sum(aux_list) // len(aux_list))
      aux_list = []

  y_coords.append(sum(aux_list) // len(aux_list))

  return y_coords

# in this form it works for proper, and only slightly warped images of scores,
# scores warped like warp_sim.png are a bit too much for this detector to handle
def find_staves_positions(img_x, img_y, stride = 1, img_array = np.array):
  probe_window_size = img_x // 30
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

    # get distance between peaks
    staff_line_distance = find_staff_line_distance(peaks_list)

    # with obtained data, find positions of staves in probe window and append to list of positions
    staves_positions_list.append(get_staves_positions(staff_line_distance, peaks_list))

    # reset probe window array
    probe_window_hist_arr = np.zeros(img_y)

  return get_stave_y_coordinate(staff_line_distance, staves_positions_list), staff_line_distance
  # return staves_positions_list

def find_staves(img = Image):
  if img.mode != 'L':
    img = img.convert('L')
    print("-- Image conversion to black&white done --")

  img = enhance_image(img)
  print("-- Image enhancement done --")

  # convert image to array to help in image processing in search of staves
  img_array = tf.keras.utils.img_to_array(img)
  img_y, img_x, c = img_array.shape
  print("-- Conversion of image to array done --")

  return find_staves_positions(img_x, img_y, 7, img_array)


