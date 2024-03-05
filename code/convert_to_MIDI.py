from staff_detection import find_staves_points, find_staves_y_points
from dewarp_page import dewarp_page
from image_segmentation import crop_staves
from ML_model.model.e2e_unfolding import get_crnn_model
from ML_model.model.model_manager import LighntingE2EModelUnfolding
from torchvision import transforms
from torch import unsqueeze
import cv2
import os
from setup import DEBUG_LEVEL
import utils
from pdf2image import convert_from_path
from verovio import toolkit


'''
Broad program overview:
 ✓- load image
 ✓- find staff positions in original image
 ✓- unwarp image
 ✓- find staff positions in unwarped image
 ✓- split image to smaller chunks containing monophonic staves or grandstaves
 ✓- pipe smaller images to model to retrieve semantic information
 ✓- concatenate obtained **kern files to one if specified
 - convert **kern file to MIDI with verovio
'''

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_tensor_list(input):
    ret_list = []
    transform = transforms.ToTensor()

    for i in input:
        aux = transform(i)
        aux = unsqueeze(aux, dim=0)
        ret_list.append(aux)

    return ret_list

def get_kern(model, filepath):
    scores_images = []
    if filepath.endswith(".pdf") or filepath.endswith(".PDF"):
        scores_images = convert_from_path(filepath)
    else:
        img = cv2.imread(filepath)
        staff_detect_img = img.copy()
        img_name = os.path.basename(filepath)
        print("-- Image loading done --")
        print("-- Loaded file: ", img_name, " --")
        scores_images.append(staff_detect_img)

    krns = []
    for score_img in scores_images:
        staves_positions, staff_line_distance = find_staves_points(score_img, img_name, 2)
        if staves_positions == -1:
            # print("-- Finding position of staves failed, aborting --")
            continue
            
        if DEBUG_LEVEL >= 2:
            utils.print_list(staves_positions)

        dewarped_img = dewarp_page(score_img, img_name, staves_positions)
        staves_y_pos = find_staves_y_points(dewarped_img, staff_line_distance)

        if DEBUG_LEVEL >= 1:
            print(staves_y_pos)

        cropped_staves = crop_staves(staff_line_distance, staves_y_pos, dewarped_img)
        tensor_staves = make_tensor_list(cropped_staves)

        one_image_krn = ""
        for ts in tensor_staves:
            pred = model(ts)
            pred_krn = utils.make_krn_from_pred(pred)

            one_image_krn += pred_krn
        krns.append(one_image_krn)

    ret_kern = ""
    for k in krns:
        ret_kern += k
    
    return [ret_kern, os.path.basename(filepath).split('.')[0]]

    
def convert_files_to_MIDI(filepaths, one_file, save_path):
    kern_content = []
    filenames = []
    model = get_crnn_model(1, 15803)
    litmodel = LighntingE2EModelUnfolding.load_from_checkpoint("code\model_checkpoints\CRNN-epoch=22-val_SER=29.46.ckpt", model=model)
    litmodel.eval()

    for file in filepaths:
        aux = get_kern(litmodel, file)
        kern_content.append(aux[0])
        filenames.append(aux[1])

    if one_file:
        one_krn_str = "**kern\t**kern\n"

        for kc in kern_content:
            one_krn_str += kc

        one_krn_str += "*-\t*-"

        krn_filename = filenames[0] + ".krn"
        save_file_name = os.path.join(save_path, krn_filename)

        with open (save_file_name, 'w') as savefile:
            savefile.write(one_krn_str)
    else:
        out_krn_content = []
        krn_start_str = "**kern\t**kern\n"
        krn_end_str = "*-\t*-"
        aux_str = ""

        for kc in kern_content:
            aux_str += krn_start_str
            aux_str += kc
            aux_str += krn_end_str
            out_krn_content.append(aux_str)
            aux_str = ""

        i = 0
        while i < len(out_krn_content):
            krn_filename = filenames[i] + ".krn"
            save_file_name = os.path.join(save_path, krn_filename)

            with open (save_file_name, 'w') as savefile:
                savefile.write(out_krn_content[i])

            i += 1

    # midi_content = toolkit.convertHumdrumToMIDI()
    