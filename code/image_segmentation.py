import cv2
from utils import resize_with_aspect_ratio
from setup import DEBUG_LEVEL


def get_distance_between(y_1, y_2):
    dist = ((y_2 - y_1) // 2)
    return y_1 + int(dist * 1.2)


def find_monophonic_boxes(x, y, staff_line_distance, staves_y_pos = []):
    mono_bb_list = []
    staves_list_len = len(staves_y_pos)
    margin = staff_line_distance * 7

    upper = staves_y_pos[0] - margin
    upper = upper if upper > 0 else 0
    
    if staves_list_len == 1:
        lower = staves_y_pos[0] + margin
        mono_bb_list.append((0, upper, x-1, lower))
    else:
        lower = get_distance_between(staves_y_pos[0], staves_y_pos[1])
        mono_bb_list.append((0, upper, x-1, lower))
        
        for i in range(1, staves_list_len-1):
            upper = get_distance_between(staves_y_pos[i-1], staves_y_pos[i])
            lower = get_distance_between(staves_y_pos[i], staves_y_pos[i+1])
            mono_bb_list.append((0, upper, x-1, lower))
        
        upper = get_distance_between(staves_y_pos[staves_list_len-2], staves_y_pos[staves_list_len-1])
        lower = staves_y_pos[staves_list_len-1] + margin
        lower = lower if lower < y else y

        mono_bb_list.append((0, upper, x-1, lower))

    return mono_bb_list


def find_grandstaff_boxes(x, y, staff_line_distance, staves_y_pos = []):
    grandstaff_bb_list = []
    staves_list_len = len(staves_y_pos)
    detected_lone_staff = False

    if staves_list_len % 2 != 0:
        staves_list_len -= 1
        detected_lone_staff = True

    margin = staff_line_distance * 7

    upper = staves_y_pos[0] - margin
    upper = upper if upper > 0 else 0


    if staves_list_len == 2:
        lower = staves_y_pos[1] + margin
        grandstaff_bb_list.append((0, upper, x-1, lower))
    else:
        lower = get_distance_between(staves_y_pos[1], staves_y_pos[2])
        grandstaff_bb_list.append((0, upper, x-1, lower))
        
        for i in range(2, staves_list_len-2, 2):
            upper = get_distance_between(staves_y_pos[i-1], staves_y_pos[i])
            lower = get_distance_between(staves_y_pos[i+1], staves_y_pos[i+2])
            grandstaff_bb_list.append((0, upper, x-1, lower))
        
        upper = get_distance_between(staves_y_pos[staves_list_len-3], staves_y_pos[staves_list_len-2])
        lower = staves_y_pos[staves_list_len-1] + margin
        lower = lower if lower < y else y

        grandstaff_bb_list.append((0, upper, x-1, lower))

    if detected_lone_staff:
        grandstaff_bb_list += find_monophonic_boxes(x, y, staff_line_distance, staves_y_pos[-1:])

    return grandstaff_bb_list


def find_bounding_boxes(x, y, staff_line_distance = int, staves_y_pos = [], grandstaff = True):
    if len(staves_y_pos) == 0 and not grandstaff:
        return []
    if len(staves_y_pos) <= 2 and grandstaff:
        return []
    
    bb_list = []

    if grandstaff:
        bb_list = find_grandstaff_boxes(x, y, staff_line_distance, staves_y_pos)
    else:
        bb_list = find_monophonic_boxes(x, y, staff_line_distance, staves_y_pos)
    
    return bb_list


def crop_staves(staff_line_distance, staves_y_pos, img, grandstaff = False):
    print("-- Dewarped image segmentation ", end="")
    y, x = img.shape[:2]
    bb_list = find_bounding_boxes(x, y, staff_line_distance, staves_y_pos, grandstaff)

    cropped_img_list = []
    for i in bb_list:
        if DEBUG_LEVEL >= 2:
           print(i)

        left = i[0]
        top = i[1]
        right = i[2]
        bottom = i[3]
        crop = img[top:bottom, left:right]
        resized = resize_with_aspect_ratio(crop, height=256, inter=cv2.INTER_LINEAR)
        rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
        cropped_img_list.append(rotated)

    if DEBUG_LEVEL >= 2:
        for i in cropped_img_list:
            resized = resize_with_aspect_ratio(i, height=1280)
            cv2.imshow("cropped", resized)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    print("done --")
    return cropped_img_list        
