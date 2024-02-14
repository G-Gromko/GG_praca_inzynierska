import numpy as np
from matplotlib import pyplot as plt
import cv2
from setup import DEBUG_LEVEL
from statistics import mode
import utils

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



def enhance_image(img, img_name):
    # image enhancer to help analize images

    # if image is RGB, convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    if DEBUG_LEVEL >= 3:
        utils.debug_show(img_name, 0.1, "blurred", img)
        
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 55, 7)

    if DEBUG_LEVEL >= 3:
        utils.debug_show(img_name, 0.2, "thresholded", img)

    img = cv2.erode(img, (5, 5), 3)

    if DEBUG_LEVEL >= 3:
        utils.debug_show(img_name, 0.3, "eroded", img)

    if DEBUG_LEVEL >= 1:
        utils.debug_show(img_name, 1, "enhanced", img)

    return img


def find_staff_line_distance_in_probe(peaks_list = []):
    space_between_peaks_list = []

    for i in range(1, len(peaks_list)):
        space_between_peaks_list.append(peaks_list[i] - peaks_list[i-1])

    if len(space_between_peaks_list) == 0:
        return 0
  
    return mode(space_between_peaks_list)

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
def filter_out_peaks(max_val, hist_arr = np.array):
    cutoff = int(max_val * 0.65)
    peak_idx_list = []

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


def get_staves_positions(x_coord, distance, lines_idx_list = []):
    if len(lines_idx_list) < 5:
        return []

    staff_line_positions = []
    counter = 0
    aux_dist = lines_idx_list[1] - lines_idx_list[0]
    flag = aux_dist >= distance - 4 and aux_dist <= distance + 4

    # distance between lines is checked with some slack to help with non-uniform filtering 
    for i in range(1, len(lines_idx_list)):
        current_dist = lines_idx_list[i] - lines_idx_list[i-1]

        if flag and counter == 4:
            counter = 0
            flag = False
            staff_line_positions.append([x_coord, (lines_idx_list[i-1] + lines_idx_list[i-5]) // 2])
        elif current_dist >= distance - 2 and current_dist <= distance + 2 and not flag:
            counter = 1
            flag = True
        elif current_dist >= distance - 2 and current_dist <= distance + 2 and flag:
            counter += 1
        elif current_dist < distance - 2 or current_dist > distance + 2:
            flag = False
            counter = 0

    if flag and counter == 4:
        staff_line_positions.append([x_coord, (lines_idx_list[i] + lines_idx_list[i-4]) // 2])

    return staff_line_positions


def make_staves_points_list(in_list, distance):
    if len(in_list) == 0:
        return -1
    elif sum([len(i) for i in in_list]) == 0:
        return -1

    staves_points_list = []
    distorted_dist = distance // 2

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
            if len(aux_list) < 3:
                aux_list = []
            else:
                staves_points_list.append(aux_list)
                aux_list = []

        
        aux_coord = i[1]
        min_dev = aux_coord - distorted_dist
        max_dev = aux_coord + distorted_dist

    for i in staves_points_list:
        i.sort(key=lambda point: point[0])

    return staves_points_list


def make_histogram_array(i, img_y, probe_window_size, img_array, probe_window_hist_arr):
    for k in range(0, img_y-1):
        for j in range(i, i + probe_window_size):
            if img_array[k][j] < 200:
                probe_window_hist_arr[k] += 1

# analizing slices of 1/40th of the image width to find staff line distance for further analisis of the image
def find_staff_line_distance(img = np.array):
    img_y, img_x = img.shape[:2]
    probe_window_size = img_x // 40
    probe_window_start_idx = img_x // 5
    probe_window_end_idx = img_x - probe_window_start_idx
    probe_window_hist_arr = np.zeros(img_y)
    staff_line_distance_list = []
    # iterate probe window over image, create histograms and get from them positions of staves
    for i in range(probe_window_start_idx, probe_window_end_idx, probe_window_size*6):

        make_histogram_array(i, img_y, probe_window_size, img, probe_window_hist_arr)

        if DEBUG_LEVEL >= 3:
            plt.plot(probe_window_hist_arr)
            title_str = "Probe window histogram of black pixels in rows from position x:{} to {}".format(i, (i + probe_window_size))
            plt.title(title_str)
            plt.ylabel("Number of dark pixels in row")
            plt.xlabel("Position on Y axis of the image")
            plt.show()

        peaks_list = filter_out_peaks(probe_window_size, probe_window_hist_arr)

        if DEBUG_LEVEL >= 4:
            print("PL: ", end='')
            print(peaks_list)

        # get distance between peaks
        staff_line_distance = find_staff_line_distance_in_probe(peaks_list)
        if staff_line_distance != 0:
            staff_line_distance_list.append(staff_line_distance)

        probe_window_hist_arr = np.zeros(img_y)

    if len(staff_line_distance_list) != 0:
        return sum(staff_line_distance_list) // len(staff_line_distance_list)
    else:
        return -1
  

def find_staves_positions(staff_line_distance, stride = 1, img = np.array):
    img_y, img_x = img.shape[:2]
    probe_window_size = (staff_line_distance * 2) + 5
    # offset for first probe window position to avoid unnecessary analizing of blank space at the edges of the sheet 
    probe_window_start_idx = probe_window_size * 2
    probe_window_end_idx = img_x - probe_window_start_idx
    probe_window_hist_arr = np.zeros(img_y)
    staves_positions_list = []
    # iterate probe window over image, create histograms and get from them positions of staves
    for i in range(probe_window_start_idx, probe_window_end_idx, probe_window_size * stride):

        make_histogram_array(i, img_y, probe_window_size, img, probe_window_hist_arr)

        # filter obtained histogram data to single positions of lines
        peaks_list = filter_out_peaks(probe_window_size, probe_window_hist_arr)

        # with obtained data, find positions of staves in probe window and append to list of positions
        staves_positions = get_staves_positions(i, staff_line_distance, peaks_list)
        staves_positions_list.append(staves_positions)

        if DEBUG_LEVEL >= 2:
            utils.print_position_list(staves_positions, i)

        if DEBUG_LEVEL >= 3:
            plt.plot(probe_window_hist_arr)
            title_str = "Probe window histogram of black pixels in rows from x-position: {} to {}".format(i, (i + probe_window_size))
            plt.title(title_str)
            plt.ylabel("Number of dark pixels in row")
            plt.xlabel("Position on Y axis of the image")
            plt.show()

        # reset probe window array
        probe_window_hist_arr = np.zeros(img_y)

    if len(staves_positions_list) == 0:
        return -1
    else:
        return staves_positions_list
  

def find_staves_points(img, img_name, stride = 2):
    print("-- Image enhancement ", end="")
    img = enhance_image(img, img_name)
    print("done --")

    print("-- Finding staff line distance ", end="")
    staff_line_dist = find_staff_line_distance(img)
    if staff_line_dist == -1:
        return -1, 0
    print("done --")
    print("-- Staff line distance: ", staff_line_dist)

    print("-- Finding positions of staves ", end="")
    staves_positions = find_staves_positions(staff_line_dist, stride, img)
    if len(staves_positions) == 0:
        return -1, -1
    else:
        print("done --")
        print("-- Reformatting of staves position list ", end= "")
        staves_points_list = make_staves_points_list(staves_positions, staff_line_dist)
        if staves_points_list == -1  or len(staves_points_list) == 0:
            return -1, -1
        print("done --")
        print("-- Got ", len(staves_positions), " staves with ", end="")
        print(sum([len(pts) for pts in staves_positions]), " points")

        return staves_points_list, staff_line_dist
    


def get_staves_y_coordinate(in_list = []):
    y_coords = []

    for stave in in_list:
        aux_sum = sum(y for x, y in stave)
        aux_y = aux_sum // len(stave)

        y_coords.append(aux_y)

    if len(y_coords) == 0:
        return -1
    
    return y_coords


def find_staves_y_points(img, staff_line_dist):

    # if image is RGB, convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    print("-- Finding positions of staves in dewarped image ", end="")
    staves_positions = find_staves_positions(staff_line_dist, 5, img)
    if len(staves_positions) == 0:
        return -1, -1
    else:
        print("done --")
        print("-- Reformatting of staves position list ", end= "")
        staves_points_list = make_staves_points_list(staves_positions, staff_line_dist)
        if staves_points_list == -1  or len(staves_points_list) == 0:
            return -1, -1
        
        print("done --")
        print("-- Getting y-coordinates of staves in dewarped image ", end="")
        y_coords = get_staves_y_coordinate(staves_points_list)
        if y_coords == -1:
            return -1, -1
        
        print("done --")
        return y_coords
