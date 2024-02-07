from staff_detection import find_staves_points, find_staves_y_points
from dewarp_page import dewarp_page
from image_segmentation import crop_staves
import cv2
import os
from setup import DEBUG_LEVEL


'''
Broad program overview:
 ✓- load image
 ✓- unwarp image (somewhat done)
 ✓- find staff positions in unwarped image
 ✓- split image to smaller chunks containing monophonic staves or grandstaves
 - pipe smaller images to model to retrieve semantic information
 - concatenate obtained **kern files to one
 - convert **kern file to MIDI with verovio
'''

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def print_list(list):
  for i in list:
    print(i)


def __main__():
  # example files for testing
  file = "test_images/DSC_0920.JPG" # Photo of music sheet on flat surface
  # file = "test_images/DSC_0921.JPG" # Photo of music sheet on flat surface
  # file = "test_images/DSC_0925.JPG" # Photo of music sheet on flat surface
  # file = "test_images/DSC_0926.JPG" # Photo of music sheet bent to book-like form


  img = cv2.imread(file)
  staff_detect_img = img.copy()
  img_name = os.path.basename(file)
  print("-- Image loading done --")
  print("-- Loaded image: ", img_name, " --")

  staves_positions, staff_line_distance = find_staves_points(staff_detect_img, img_name, 2)
  if staves_positions == -1:

    print("-- First attempt failed, trying alternative mode of finding staves --")
    staves_positions, staff_line_distance = find_staves_points(staff_detect_img, img_name, 2, True)

    if staves_positions == -1:
      print("-- Second attempt failed, aborting --")
      return
    
  if DEBUG_LEVEL >= 2:
    print_list(staves_positions)

  dewarped_img = dewarp_page(img, img_name, staves_positions)
  staves_y_pos = find_staves_y_points(dewarped_img, staff_line_distance)

  print(staves_y_pos)

  cropped_staves = crop_staves(staff_line_distance, staves_y_pos, dewarped_img, False)
  # print("-- Obtaining semantic information done --")
  # print("-- Concatenation of semantic information done --")
  # print("-- Conversion to MIDI done --")
  # print("-- Saving obtained MIDI files done --")

if __name__ == "__main__":
  __main__()