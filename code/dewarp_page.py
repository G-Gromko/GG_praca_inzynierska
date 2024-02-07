import datetime
import cv2
import numpy as np
import scipy.optimize
from setup import DEBUG_LEVEL
from utils import debug_show

ROTATION_VECTOR_IDX = slice(0, 3)   # index of rvec in params vector
TRANSLATION_VECTOR_IDX = slice(3, 6)   # index of translation_vector in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector
FOCAL_LENGTH = 1.2       # normalized focal length of camera

ADA_THRESH_WIN_SZ = 55      # window size for adaptive threshold in reduced px

OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16      # downscaling factor for remapping image

# default intrinsic parameter matrix
KAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)


def pix2norm(shape, points):
    height, width = shape[:2]
    scale = 2.0/(max(height, width))
    offset = np.array([width, height], dtype=points.dtype).reshape((-1, 1, 2))*0.5
    return (points - offset) * scale


def norm2pix(shape, points, as_integer):
    height, width = shape[:2]
    scale = max(height, width)*0.5
    offset = np.array([0.5*width, 0.5*height], dtype=points.dtype).reshape((-1, 1, 2))
    rval = points * scale + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval
    

def project_keypoints(pvec, keypoint_index):

    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0

    return project_xy(xy_coords, pvec)


def draw_correspondences(img, dstpoints, projpts):

    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)),
                       (dstpoints, (0, 0, 255))]:

        for point in pts:
            cv2.circle(display, fltp(point), 15, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 10, cv2.LINE_AA)

    return display
    

def get_page_extents(image):

    height, width = image.shape[:2]

    margin = width // 25

    xmin = margin
    ymin = margin
    xmax = width-margin
    ymax = height-margin

    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline


def fltp(point):
    return tuple(point.astype(int).flatten())


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem
    

def get_default_params(corners, ycoords, xcoords):

    # page width and height
    page_width = np.linalg.norm(corners[1] - corners[0])
    page_height = np.linalg.norm(corners[-1] - corners[0])
    rough_dimensions = (page_width, page_height)

    # our initial guess for the cubic has no slope
    cubic_slopes = [0.0, 0.0]

    # object points of flat page in 3D coordinates
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]])

    # estimate rotation and translation from four 2D-to-3D point
    # correspondences
    _, roatation_vector, translation_vector = cv2.solvePnP(corners_object3d, corners, KAMERA_MATRIX, np.zeros(5))

    span_counts = [len(xc) for xc in xcoords]

    params = np.hstack((np.array(roatation_vector).flatten(),
                        np.array(translation_vector).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) + tuple(xcoords))

    return rough_dimensions, span_counts, params


def project_xy(xy_coords, parameter_vector):

    # get cubic polynomial coefficients given
    #
    #  f(0) = 0, f'(0) = alpha
    #  f(1) = 0, f'(1) = beta

    alpha, beta = tuple(parameter_vector[CUBIC_IDX])

    poly = np.array([ alpha + beta, -2*alpha - beta, alpha, 0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

    image_points, _ = cv2.projectPoints(objpoints, parameter_vector[ROTATION_VECTOR_IDX], parameter_vector[TRANSLATION_VECTOR_IDX], KAMERA_MATRIX, np.zeros(5))

    return image_points


def project_keypoints(parameter_vector, keypoint_index):

    xy_coords = parameter_vector[keypoint_index]
    xy_coords[0, :] = 0

    return project_xy(xy_coords, parameter_vector)


def keypoints_from_samples(name, image, pagemask, page_outline, staff_points):

    all_eigen_vectors = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in staff_points:

        _, eigen_vector = cv2.PCACompute(points.reshape((-1, 2)), None, maxComponents=1)

        weight = np.linalg.norm(points[-1] - points[0])

        all_eigen_vectors += eigen_vector * weight
        all_weights += weight

    eigen_vector = all_eigen_vectors / all_weights

    x_dir = eigen_vector.flatten()

    if x_dir[0] < 0:
        x_dir = -x_dir

    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)

    px0 = px_coords.min()
    px1 = px_coords.max()

    py0 = py_coords.min()
    py1 = py_coords.max()

    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir

    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

    ycoords = []
    xcoords = []

    for points in staff_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)

    return corners, np.array(ycoords), xcoords


def make_keypoint_index(span_counts):

    no_of_spans = len(span_counts)
    no_of_points = sum(span_counts)
    keypoint_index = np.zeros((no_of_points+1, 2), dtype=int)
    start = 1

    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start+end, 1] = 8+i
        start = end

    keypoint_index[1:, 0] = np.arange(no_of_points) + 8 + no_of_spans

    return keypoint_index


def optimize_params(name, small, destination_points, span_counts, params):

    keypoint_index = make_keypoint_index(span_counts)

    def objective(parameter_vector):
        parameter_points = project_keypoints(parameter_vector, keypoint_index)
        return np.sum((destination_points - parameter_points)**2)

    if DEBUG_LEVEL >= 1:
        print('  initial objective is', objective(params))

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, destination_points, projpts)
        debug_show(name, 4, 'keypoints before', display)

    if DEBUG_LEVEL >= 1:
        print('  optimizing', len(params), 'parameters...')
    start = datetime.datetime.now()
    result = scipy.optimize.minimize(objective, params,
                                  method='Powell')
    end = datetime.datetime.now()
    if DEBUG_LEVEL >= 1:
        print('  optimization took', round((end-start).total_seconds(), 2), 'sec.')
        print('  final objective is', result.fun)
    params = result.x

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, destination_points, projpts)
        debug_show(name, 5, 'keypoints after', display)

    return params


def get_page_dimensions(corners, rough_dimensions, params):

    dst_br = corners[2].flatten()

    dimensions = np.array(rough_dimensions)

    def objective(dimensions):
        proj_br = project_xy(dimensions, params)
        return np.sum((dst_br - proj_br.flatten())**2)

    result = scipy.optimize.minimize(objective, dimensions, method='Powell')
    dimensions = result.x

    if DEBUG_LEVEL >= 1:
        print('  got page dimensions', dimensions[0], 'x', dimensions[1])

    return dimensions


def remap_image(name, img, page_dimensions, params):

    height = 0.5 * page_dimensions[1] * OUTPUT_ZOOM * img.shape[0]
    height = round_nearest_multiple(height, REMAP_DECIMATE)

    width = round_nearest_multiple(height * page_dimensions[0] / page_dimensions[1], REMAP_DECIMATE)

    if DEBUG_LEVEL >= 1:
        print('  output will be {}x{}'.format(width, height))

    height_small = height // REMAP_DECIMATE
    width_small = width // REMAP_DECIMATE

    page_x_range = np.linspace(0, page_dimensions[0], width_small)
    page_y_range = np.linspace(0, page_dimensions[1], height_small)

    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

    page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                page_y_coords.flatten().reshape((-1, 1))))

    page_xy_coords = page_xy_coords.astype(np.float32)

    image_points = project_xy(page_xy_coords, params)
    image_points = norm2pix(img.shape, image_points, False)

    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

    image_x_coords = cv2.resize(image_x_coords, (width, height), interpolation=cv2.INTER_CUBIC)

    image_y_coords = cv2.resize(image_y_coords, (width, height), interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    remapped = cv2.remap(img_gray, image_x_coords, image_y_coords, 
                         cv2.INTER_CUBIC, None, cv2.BORDER_REPLICATE)

    thresh = cv2.adaptiveThreshold(remapped, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, ADA_THRESH_WIN_SZ, 25)

    return thresh


def dewarp_page(img, img_name, staves_positions_in_list):
    print("-- Page dewarping", end="")

    shape = img.shape[:2]

    staves_positions = []

    for stave_pos in staves_positions_in_list:
        aux = np.array(stave_pos, dtype=np.float32).reshape(-1, 1, 2)
        aux = pix2norm(shape, aux)
        staves_positions.append(aux)

    pagemask, page_outline = get_page_extents(img)

    corners, y_coords, x_coords = keypoints_from_samples(img_name, img, pagemask, page_outline, staves_positions)

    rough_dimensions, span_counts, params = get_default_params(corners, y_coords, x_coords)

    destination_points = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(staves_positions))

    params = optimize_params(img_name, img, destination_points, span_counts, params)

    page_dimensions = get_page_dimensions(corners, rough_dimensions, params)

    outfile = remap_image(img_name, img, page_dimensions, params)

    print(" done --")

    return outfile