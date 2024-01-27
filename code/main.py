from PIL import Image
from staff_detection import find_staves
from dewarp_page import dewarp_page
from image_segmentation import crop_staves


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

def __main__():
  # example files for testing
  # file = "test_images/lg-97012964-aug-lilyjazz--page-4.png" # example image of proper music score sheet from DeepScore2 dataset
  # file = "test_images/maj3_down_m-4-7_distorted.jpg" # example image of distorted staves from Grandstaff dataset
  # file = "test_images/warp_sim.png" # simulated distorted image of music score
  file = "test_images/mora_the_spider.png"

  img = Image.open(file)
  print("-- Image loading done --")
  
  img = dewarp_page(img) # <-- placeholder function

  # convert original image to RGB bc model was trained on RGB images
  if img.mode != 'RGB':
    img = img.convert('RGB')

  staves_positions, staff_line_distance = find_staves(img)
  print("-- Finding positions of staves done --")

  print(staves_positions, " ", staff_line_distance)
  cropped_staves = crop_staves(staff_line_distance, staves_positions, img, 2)
  print("-- Segmentation done --")

  # print("-- Unwarping done --")
  # print("-- Obtaining semantic information done --")
  # print("-- Concatenation of semantic information done --")
  # print("-- Conversion to MIDI done --")
  # print("-- Saving obtained MIDI files done --")

if __name__ == "__main__":
  __main__()