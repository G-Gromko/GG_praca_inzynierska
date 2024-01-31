from PIL import Image
from staff_detection import find_staves
from dewarp_page import dewarp_page
from image_segmentation import crop_staves
import cv2
import os
from setup import DEBUG_LEVEL


'''
Broad program overview:
 ✓- load image
 - unwarp image
 ✓- find staff positions in unwarped image
 ✓- split image to smaller chunks containing desired number of staves (e.g. monophonic score should have one, pianoform score should have two)
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
  # file = "test_images/DSC_0920.JPG"
  # file = "test_images/DSC_0921.JPG"
  file = "test_images/DSC_0924.JPG"


  img = cv2.imread(file, 0)
  print("-- Image loading done --")
  print("-- Loaded image: ", os.path.basename(file), " --")

  staves_positions, staff_line_distance = find_staves(img, 3)
  if staves_positions == -1:
    print("-- Something went wrong, terminating --")
    return
  if DEBUG_LEVEL >= 1:
    print_list(staves_positions)
    
  print("-- Finding positions of staves done --")
  print("-- Got ", len(staves_positions), " staves with ", end="")
  print(sum([len(pts) for pts in staves_positions]), " points")

  # img = dewarp_page(img, staves_positions) # <-- placeholder function

  # print(staves_positions, " ", staff_line_distance)
  # cropped_staves = crop_staves(staff_line_distance, staves_positions, img, 2)
  # print("-- Segmentation done --")

  # print("-- Unwarping done --")
  # print("-- Obtaining semantic information done --")
  # print("-- Concatenation of semantic information done --")
  # print("-- Conversion to MIDI done --")
  # print("-- Saving obtained MIDI files done --")

if __name__ == "__main__":
  __main__()