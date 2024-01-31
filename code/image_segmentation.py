from PIL import Image

def find_bounding_boxes(x, y, staff_line_distance = int, staves_y_pos = [], no_of_staves = 1):
  distance = ((staff_line_distance * 4) + 5) * 2.2
  bb_list = []
  staves_list_len = len(staves_y_pos)
  upper = lower = 0
  
  i = 0
  while i < staves_list_len:
    upper = staves_y_pos[i] - distance
    upper = upper if upper > 0 else 0

    i += (no_of_staves - 1)
    if i >= staves_list_len:
      lower = staves_y_pos[i-1] + distance
      lower = lower if lower < y else y-1
      break
    else:
      lower = staves_y_pos[i] + distance
      lower = lower if lower < y else y-1

    bb_list.append((0, upper, x-1, lower))
    i += 1
    
  return bb_list

def crop_staves(staff_line_distance, staves_y_pos = [], img = Image, no_of_staves = 1):
  x, y = img.size
  bb_list = find_bounding_boxes(x, y, staff_line_distance, staves_y_pos, no_of_staves)

  cropped_img_list = []
  for i in bb_list:
    cropped_img_list.append(img.crop((i)))

  for i in cropped_img_list:
    i.show()