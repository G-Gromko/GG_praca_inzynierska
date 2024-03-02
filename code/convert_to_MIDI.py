from staff_detection import find_staves_points, find_staves_y_points
from dewarp_page import dewarp_page
from image_segmentation import crop_staves
import cv2
import os
from setup import DEBUG_LEVEL
import utils


'''
Broad program overview:
 ✓- load image
 ✓- find staff positions in original image
 ✓- unwarp image
 ✓- find staff positions in unwarped image
 ✓- split image to smaller chunks containing monophonic staves or grandstaves
 - pipe smaller images to model to retrieve semantic information
 - concatenate obtained **kern files to one
 - convert **kern file to MIDI with verovio
'''

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_kern(filepath):
    # example files for testing
    # filepath = "test_images/DSC_0920.JPG" # Photo of music sheet on flat surface
    # filepath = "test_images/DSC_0921.JPG" # Photo of music sheet on flat surface
    # filepath = "test_images/DSC_0925.JPG" # Photo of music sheet on flat surface
    # filepath = "test_images/DSC_0926.JPG" # Photo of music sheet bent to book-like form


    img = cv2.imread(filepath)
    staff_detect_img = img.copy()
    img_name = os.path.basename(filepath)
    print("-- Image loading done --")
    print("-- Loaded image: ", img_name, " --")

    staves_positions, staff_line_distance = find_staves_points(staff_detect_img, img_name, 2)
    if staves_positions == -1:
        print("-- Finding position of staves failed, aborting --")
        
    if DEBUG_LEVEL >= 2:
        utils.print_list(staves_positions)

    dewarped_img = dewarp_page(img, img_name, staves_positions)
    staves_y_pos = find_staves_y_points(dewarped_img, staff_line_distance)

    print(staves_y_pos)

    cropped_staves = crop_staves(staff_line_distance, staves_y_pos, dewarped_img)
    # print("-- Obtaining semantic information done --")
    # print("-- Concatenation of semantic information done --")
    # print("-- Conversion to MIDI done --")
    # print("-- Saving obtained MIDI files done --")

    
def convert_files_to_MIDI(filepaths=[], make_pdf = False, one_file = False):
    kern_files = []

    for file in filepaths:
        kern_files.append(get_kern(file))

    # if one_file:
        # prepare_kern_files(kern_files)

    