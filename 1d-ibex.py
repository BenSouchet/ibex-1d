
import os
import sys
import math
import shutil
import logging
import numpy
from numpy.typing import NDArray
import cv2 as cv
from pathlib import Path
from datetime import datetime

SCRIPT_NAME = '1D-IBEX : 1D Image Barcode EXtractor'
VERSION = '1.0.0'

RESULT_IMAGE_EXT = '.png' # Can be any type handled by OpenCV, see documentation for valid values.
DETECTION_IMAGE_MAX_DIM = 1024 # In pixels, if the lagest dimension (width or height) of the input image is
#                                bigger than this value the image will be downscale ONLY for paper detect calculations.
#                                Smaller value mean faster computation but less accuracy.
MAX_CONTOUR_AREA = 0. # Set directly by the function 
MIN_CONTOUR_PERIMETER = DETECTION_IMAGE_MAX_DIM * .05 # Contours perimeters below this value will be skipped.
MIN_BARCODE_RECT_RATIO = 4. # Minimum ratio to consider a rectangle to be part of a barcode
SIMPLIFIED_CONTOUR_MAX_COEF = .15 # Maximum ratio of simplification allowed for the contour point reduction (e.g. simplify_contour function)
MIN_SPLIT_BARECODE = 70 # Minium number of columns in a barecode
MIN_BAR_NUMBER = 18 # Minium number of bars in a group to consider it a barcode
BAR_DETECTION_FACTOR = 4. # A multiplication factor used for the radius to detect bars of the same barcode
SCALE_DETECTED_BARCODE_FACTOR = 1.04 # A scale value superior to 1.0 to add an offset to detected barcode shape
BARCODE_DEFORMATION_TOLERANCE = 0.01 # Above this value a complexe method will be used to compute paper aspect ratio

DEBUG = False

class ColoredFormatter(logging.Formatter):
    """Custom formatter handling color"""
    cyan = '\x1b[36;20m'
    green = '\x1b[32;20m'
    yellow = '\x1b[33;20m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'
    message_format = '%(levelname)-8s - %(message)s'

    FORMATS = {
        logging.DEBUG: cyan + message_format + reset,
        logging.INFO: green + message_format + reset,
        logging.WARNING: yellow + message_format + reset,
        logging.ERROR: red + message_format + reset,
        logging.CRITICAL: bold_red + message_format + reset
    }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)

def init_logger()-> logging.Logger:
    """Initialize script logger"""

    logger_name = Path(__file__).stem
    logger = logging.getLogger(logger_name)
    logger.setLevel(level = logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=logging.DEBUG)
    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(console_handler)

    return logger

LOG = init_logger()

def delete_results_folder()-> bool:
    if PATH_DIR_RESULTS and PATH_DIR_RESULTS.exists():
        try:
            shutil.rmtree(PATH_DIR_RESULTS)
        except Exception as e:
            LOG.error('Failed to delete the empty results folder. Reason: {}'.format(e))
            return False
    return True

def create_results_folder()-> bool:
    working_directory = Path.cwd()

    # Step 1: Create the directory path for the results and set it as a global variable
    global PATH_DIR_RESULTS
    PATH_DIR_RESULTS = working_directory.joinpath('results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Step 2: Creating the directories
    try:
        PATH_DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        LOG.error('Failed to create results folder. Reason: {}'.format(e))
        return False

    return True

def save_to_results_folder(image: NDArray[numpy.uint8], filename: str):
    cv.imwrite(str(PATH_DIR_RESULTS.joinpath(filename + RESULT_IMAGE_EXT)), image)

def downscale_image(image: NDArray[numpy.uint8])-> tuple[float, NDArray[numpy.uint8]]:
    factor = 1.0
    height, width = image.shape[:2]

    # Step 1: If image doesn't need resize do nothing
    if height <= DETECTION_IMAGE_MAX_DIM and width <= DETECTION_IMAGE_MAX_DIM:
        return (1.0, image)

    # Step 2: Determine the biggest dimension between height and width
    if height > width:
        # Step 3: Compute the new dimension, scaling by reduction factor
        factor = (float(DETECTION_IMAGE_MAX_DIM) / float(height))
        width = int(float(width) * factor)
        height = DETECTION_IMAGE_MAX_DIM
    else:
        # Step 3: Compute the new dimension, scaling by reduction factor
        factor = (float(DETECTION_IMAGE_MAX_DIM) / float(width))
        height = int(float(height) * factor)
        width = DETECTION_IMAGE_MAX_DIM

    # Step 4: Resize and return the new image
    return (factor, cv.resize(image, (width, height), interpolation=cv.INTER_AREA))

def split_grayscale(image: NDArray[numpy.uint8])-> tuple[float, NDArray[numpy.uint8]]:
    if len(image.shape) >= 3:
        YUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        return (YUV[:, :, 0], YUV)
    return (image, image)

def simplify_contour_compute_weight(contour: NDArray[numpy.float32], index: int)-> float:
    p1 = contour[(index-1)%contour.shape[0]][0]
    p2 = contour[index][0]
    p3 = contour[(index+1)%contour.shape[0]][0]
    return (0.5 * abs((p1[0] * (p2[1] - p3[1])) + (p2[0] * (p3[1] - p1[1])) + (p3[0] * (p1[1] - p2[1]))))

def simplify_contour(contour: NDArray[numpy.float32], nbr_ptr_limit: int = 4)-> NDArray[numpy.float32]:
    # Using a naive version of Visvalingam-Whyatt simplification algorithm

    # points_weights will be used to determine the importance of points,
    # in the Visvalingam-Whyatt algorithm it's the area of the triangle created by a point and his direct neighbours
    points_weights = numpy.zeros(contour.shape[0])

    # Step 1: First pass, computing all points weight
    for index in range(contour.shape[0]):
        points_weights[index] = simplify_contour_compute_weight(contour, index)

    # Step 2: Until we have 4 points we delete the less significant point and iterate
    while contour.shape[0] > nbr_ptr_limit:
        # Step 2.A: Get point index with minimum weight
        index_pnt = numpy.argmin(points_weights)

        # Step 2.B: Remove it
        contour = numpy.delete(contour, index_pnt, axis=0)
        points_weights = numpy.delete(points_weights, index_pnt)
        if contour.shape[0] == nbr_ptr_limit:
            break

        # Step 2.C: Re-compute neighbours points weight
        index_pnt_prev = (index_pnt-1)%contour.shape[0]
        index_pnt_next = (index_pnt)%contour.shape[0]
        points_weights[index_pnt_prev] = simplify_contour_compute_weight(contour, index_pnt_prev)
        points_weights[index_pnt_next] = simplify_contour_compute_weight(contour, index_pnt_next)

    return contour

def are_angles_equal(angle1: float, angle2: float, tolerance: float)-> bool:
    # Get the difference in the angles
    diff = abs(angle1 - angle2)

    # Reduce to range [0, 2*PI)
    diff = diff % (2. * numpy.pi)
    
    # Reduce to range (-PI, PI]
    if diff > numpy.pi:
        diff -= 2. * numpy.pi

    # Get the absolute difference in the range [0,PI]
    diff = abs(diff)

    return diff <= tolerance

def normalize_vector(vector):
    length = numpy.linalg.norm(vector)
    if length == 0:
       return vector
    return vector / length

def scale_contour_from_centroid(contour: NDArray[numpy.float32], scale: float, to_int: bool=False)-> NDArray[numpy.float32]:
    # Step 1: Determine the centroid of the contour
    moment = cv.moments(contour)
    center_x = int(moment['m10'] / moment['m00'])
    center_y = int(moment['m01'] / moment['m00'])

    # Step 2: move the contour center at 0,0
    contour_normalized = contour - [center_x, center_y]

    # Step 3: Scale
    contour = contour_normalized * scale

    # Step 4: Move back the contour to it position
    contour = contour + [center_x, center_y]
    
    if to_int:
        return (numpy.rint(contour)).astype(int)

    return contour

class BarCandidate:
    points: list[tuple[int, int]]
    dimensions: tuple[float, float]
    centroid: tuple[int, int]
    area: float
    angle: float
    detection_radius: float

    def set_points_from_contour(self, contour: NDArray[numpy.float32], debug:bool = False):
        # Step 1: Find top left point, using distance to top left of the picture
        dist_list = [[numpy.linalg.norm(point), index] for index, point in enumerate(contour)]
        dist_list = sorted(dist_list, key = lambda x: x[0])

        index_pnt_tl = dist_list[0][1]

        # Step 2: Find the others points order. Since the contour has been retrieved via 
        #         cv.findContours it's either sorted in clockwise or counter clockwise,
        count_points = 4# We know at this point that the contour as only 4 points, no more, no less
        index_pnt_prev = (index_pnt_tl-1)%count_points
        index_pnt_next = (index_pnt_tl+1)%count_points
        index_pnt_last = (index_pnt_tl+2)%count_points

        # Step 2.B: Comparing x axis values of the neighbours of the top left point find out if the
        #           contour has been sorted in clockwise or counter clockwise
        if contour[index_pnt_prev][0] > contour[index_pnt_next][0]:
            # Counter clockwise
            self.points = [ contour[index_pnt_tl], contour[index_pnt_prev],
                            contour[index_pnt_last], contour[index_pnt_next] ]
        else:
            # Clockwise
            self.points = [ contour[index_pnt_tl], contour[index_pnt_next],
                            contour[index_pnt_last], contour[index_pnt_prev] ]

    def compute_angle(self):
        v0 = numpy.array((0, 1))
        v1 = normalize_vector(numpy.array(self.points[3]) - numpy.array(self.points[0]))

        self.angle = numpy.arctan2(numpy.linalg.det([v0,v1]),numpy.dot(v0,v1))

    def __init__(self, contour: NDArray[numpy.float32], area: float, dimensions: tuple[float, float]):
        # Points
        self.set_points_from_contour(contour)
        # Dimensions
        self.dimensions = dimensions
        # Centroid
        M = cv.moments(contour)
        self.centroid = (round(M['m10']/M['m00']), round(M['m01']/M['m00']))
        # Area
        self.area = area
        # Angle (in radians)
        self.compute_angle()
        # Detection Radius
        self.detection_radius = min(self.dimensions[0], self.dimensions[1]) * BAR_DETECTION_FACTOR

def compute_aspect_ratio(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32])-> float:
    # Based on :
    # - https://www.microsoft.com/en-us/research/publication/2016/11/Digital-Signal-Processing.pdf
    # - http://research.microsoft.com/en-us/um/people/zhang/papers/tr03-39.pdf
    # - https://andrewkay.name/blog/post/aspect-ratio-of-a-rectangle-in-perspective/

    # Step 1: Get image center, will be used as origin
    h, w = image.shape[:2]
    origin = (w * .5, h * .5)

    # Step 2: Homeneous points coords from image origin
    # /!\ CAREFUL : points need to be in zig-zag order (A, B, D, C)
    p1 = numpy.array([*(corners[0] - origin), 1.])
    p2 = numpy.array([*(corners[1] - origin), 1.])
    p3 = numpy.array([*(corners[3] - origin), 1.])
    p4 = numpy.array([*(corners[2] - origin), 1.])

    # Step 3: Zhengyou Zhang p.10 : equations (11) & (12)
    k2 = numpy.dot(numpy.cross(p1, p4), p3) / numpy.dot(numpy.cross(p2, p4), p3)
    k3 = numpy.dot(numpy.cross(p1, p4), p2) / numpy.dot(numpy.cross(p3, p4), p2)

    # Step 4: Compute the focal length
    f = 0.
    f_sq = -((k3 * p3[1] - p1[1]) * (k2 * p2[1] - p1[1]) + \
             (k3 * p3[0] - p1[0]) * (k2 * p2[0] - p1[0]) ) / ((k3 - 1) * (k2 - 1))
    if f_sq > 0.:
        f = numpy.sqrt(f_sq)
    # If l_sq <= 0, λ cannot be computed, two sides of the rectangle's image are parallel
    # Either Uz and/or Vz is equal zero, so we leave l = 0

    # Step 5: Computing U & V vectors, BUT the z value of these vectors are in the form: z / f
    # Where f is the focal length
    u = (k2 * p2) - p1
    v = (k3 * p3) - p1

    # Step 6: Get length of U & V
    len_u = numpy.linalg.norm([u[0], u[1], (u[2] * f)])
    len_v = numpy.linalg.norm([v[0], v[1], (v[2] * f)])

    return (len_v / len_u)

def compute_barcode_size(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32])-> tuple[int, int]:
    # Vectors of the side of the contour (clockwise)
    side_top_vec = corners[1] - corners[0]
    side_rgh_vec = corners[2] - corners[1]
    side_btm_vec = corners[2] - corners[3]
    side_lft_vec = corners[3] - corners[0]

    # Step 1: Compute average width & height of the paper sheet
    paper_avg_width = 0.5 * (numpy.linalg.norm(side_top_vec) + numpy.linalg.norm(side_btm_vec))
    paper_avg_height = 0.5 * (numpy.linalg.norm(side_lft_vec) + numpy.linalg.norm(side_rgh_vec))

    # Step 2: If deformation is negligable avoid computation and return the average dimensions
    #         Checking if the opposite sides are parallel
    if math.isclose((side_top_vec[0] * side_btm_vec[1]), (side_top_vec[1] * side_btm_vec[0]), abs_tol=BARCODE_DEFORMATION_TOLERANCE) and \
        math.isclose((side_lft_vec[0] * side_rgh_vec[1]), (side_lft_vec[1] * side_rgh_vec[0]), abs_tol=BARCODE_DEFORMATION_TOLERANCE):
        return (round(paper_avg_width), round(paper_avg_height))

    # Step 3: Compute aspect ratio
    aspect_ratio = compute_aspect_ratio(image, corners)

    if aspect_ratio == 0.:
        # The ratio could not be computed, use a fallback
        rect = cv.minAreaRect(corners)
        return (rect.size.width, rect.size.height)

    return (round(paper_avg_width), round(paper_avg_width * aspect_ratio))

def extract_barcode(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32])-> NDArray[numpy.uint8]:
    # Step 1: Compute size (width & height) of the barcode
    (width, height) = compute_barcode_size(image, corners)

    # Step 2: Create the destination image size array
    dim_dest_image = numpy.array([[0., 0.], [(width - 1.), 0.], [(width - 1.), (height - 1.)], [0., (height - 1.)]])

    # Step 3: Compute the perspective deformation matrix
    #         /!\ inputs need to be numpy array in float32
    M = cv.getPerspectiveTransform(corners.astype(numpy.float32), dim_dest_image.astype(numpy.float32))

    # Step 4: Extract and unwrap/straighten the barcode
    barcode = cv.warpPerspective(image, M, (int(width), int(height)), borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return barcode

def find_image_contours(image: NDArray[numpy.uint8], use_adaptive_treshold:bool, image_index: int)-> list[NDArray[numpy.float32]]:
    # Step 1: Apply threshold to be able to detect barcode lines
    if use_adaptive_treshold:
        # Blocksize is 1/10 of the smallest of the two dimensions rounded to the nearest odd number
        height, width = image.shape[:2]
        blocksize = int(math.ceil(min(width, height) * .05) * 2 + 1)
        c_const = round(blocksize * .1)# 1/10 of blocksize
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blocksize, c_const)
    else:
        image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    if DEBUG:
        save_to_results_folder(image, 'Image-{:03d}_DEBUG_03_Treshold'.format(image_index))

    # Step 2: Find contours
    contours , _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return contours

def find_groups(contours: list[NDArray[numpy.float32]], max_contour_area:float)->list[list[BarCandidate]]:
    # Step 1: Store bars candidates
    candidates = []

    for contour in contours:
        contour = cv.convexHull(contour)

        perimeter = cv.arcLength(contour, True)
        area = cv.contourArea(contour)

        if perimeter < MIN_CONTOUR_PERIMETER or area > max_contour_area:
            continue

        rect = cv.minAreaRect(contour)
        (width, height) = rect[1]
        if (max(width, height) / max(min(width, height), 1.)) < MIN_BARCODE_RECT_RATIO:
            continue

        contour = cv.boxPoints(rect)

        candidates.append(BarCandidate(numpy.squeeze(contour), area, rect[1]))

    if not candidates:
        return None

    # Step 2: Group linked bars, approx similar angles + close to each others
    groups = []
    while len(candidates):
        group = [candidates.pop()]
        for current in group:
            for index_candidate, candidate in reversed(list(enumerate(candidates))):
                if are_angles_equal(current.angle, candidate.angle, tolerance=0.07) and \
                        numpy.linalg.norm(numpy.subtract(candidate.centroid, current.centroid)) <= max(current.detection_radius, candidate.detection_radius):
                    group.append(candidates.pop(index_candidate))
        groups.append(group)

    return groups

def get_corners_from_coutour(contour: NDArray[numpy.float32])-> NDArray[numpy.float32]:
    # We need first to ensure a clockwise orientation for the contour
    corners = None

    # Step 1: Find top left point, using distance to top left of the picture
    dist_list = [[numpy.linalg.norm(point[0]), index] for index, point in enumerate(contour)]
    dist_list = sorted(dist_list, key = lambda x: x[0])

    index_pnt_tl = dist_list[0][1]

    # Step 2: Find the others points order. Since the contour has been retrieved via 
    #         cv.findContours it's either sorted in clockwise or counter clockwise,
    count_points = 4# We know at this point that the contour as only 4 points, no more, no less
    index_pnt_prev = (index_pnt_tl-1)%count_points
    index_pnt_next = (index_pnt_tl+1)%count_points
    index_pnt_last = (index_pnt_tl+2)%count_points
    # Step 2.B: Comparing x axis values of the neighbours of the top left point find out if the
    #           contour has been sorted in clockwise or counter clockwise
    if contour[index_pnt_prev][0][0] > contour[index_pnt_next][0][0]:
        # Counter clockwise
        corners = numpy.array([contour[index_pnt_tl][0],
                                contour[index_pnt_prev][0],
                                contour[index_pnt_last][0],
                                contour[index_pnt_next][0]])
    else:
        # Clockwise
        corners = numpy.array([contour[index_pnt_tl][0],
                                contour[index_pnt_next][0],
                                contour[index_pnt_last][0],
                                contour[index_pnt_prev][0]])

    # Step 3: Convert array to int
    #corners = numpy.rint(corners).astype(int)

    return corners

def check_need_rotation(corners: NDArray[numpy.float32])-> bool:
    # Step 1: Compute vectors of edge 1 and 2 of the shape
    side_top_vec = corners[1] - corners[0]
    side_rgh_vec = corners[2] - corners[1]
    
    # Step 2: Determine if the result need a 90 degrees rotation
    #         we can do a simple test because we have ordered the the corners in clockwise
    #         and first point is the closest to top left corner of image.
    if numpy.linalg.norm(side_top_vec) < numpy.linalg.norm(side_rgh_vec):
        # First edge smaller than second, shape need the rotation
        return True
    return False

def extract_barcodes(image: NDArray[numpy.uint8], groups: list[list[BarCandidate]], scale_factor:float, image_index: int):
    barcode_index = 1

    for group in groups:
        if len(group) < MIN_BAR_NUMBER:
            # Not enought bars to consider this group valid
            continue

        # Step 1: Get the enclosing convex shape
        # Step 1.A: Append the points of all the bars of the groupAdd points of the current
        points = []
        for bar in group:
            points.extend(bar.points)

        # Step 1.B: Reshape to have same format as a contour to call convexhull OpenCV method
        temp_contour = numpy.array(points).reshape((-1,1,2)).astype(numpy.int32)
        hull = cv.convexHull(temp_contour)
        cnt_points = len(hull)

        # Step 1.C: Ensure we have a shape composed of 4 points (corners)
        if cnt_points < 4:
            continue

        if cnt_points > 4:
            approx = cv.approxPolyDP(hull, 0.02 * cv.arcLength(hull, True), True)
            cnt_points = len(approx)
            if cnt_points == 4:
                hull = approx
            else:
                hull = simplify_contour(hull)
            cnt_points = 4

        # Step 2: Scale up a bit the shape to add space
        hull = scale_contour_from_centroid(hull, SCALE_DETECTED_BARCODE_FACTOR)

        # Step 3: Very important /!\ Apply the scale factor to scale up the contours to the correct size
        hull = hull * scale_factor
        corners = get_corners_from_coutour(hull)

        if DEBUG:
            debug_image = image.copy()
            cv.drawContours(debug_image, [corners.astype(numpy.int32)], 0, (0, 0, 255), 1, cv.LINE_AA)
            save_to_results_folder(debug_image, 'Image-{:03d}_DEBUG_04_barcode{:03d}'.format(image_index, barcode_index))

        # Step 4: Extract / unwrap the barcode
        barcode_image = extract_barcode(image, corners)

        # Step 5: Ensure the barcode is properly rotated (horizontal)
        if check_need_rotation(corners):
            barcode_image = cv.rotate(barcode_image, cv.ROTATE_90_CLOCKWISE)

        # Step 6: Save to folder
        filename = "Image-{:03d}_Barcode-{:03d}".format(image_index, barcode_index)
        save_to_results_folder(barcode_image, filename)

        barcode_index += 1

def main(images_paths: list[str], use_adaptive_treshold:bool=False)-> bool:
    """Entry point"""

    # Step 1: Create the result folder
    folder_created = create_results_folder()
    if not folder_created:
        return False

    # Step 2: Extract barcodes in images
    image_index = 1
    for image_path in images_paths:
        # Step 2.A: Ensure the path exists and is a valid file
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            LOG.error('Path "{}": Doesn\'t exist or isn\'t a file.'.format(path))
            continue

        # Step 2.B: Try reading / loading the image
        image = cv.imread(str(path.resolve()))
        if image is None:
            LOG.error('Path "{}": Cannot read the image.'.format(path))
            continue

        # Step 2.C: Get grayscale (Using Y of YUV, this is better than GRAY)
        gray, _ = split_grayscale(image)
        if DEBUG:
            save_to_results_folder(gray, 'Image-{:03d}_DEBUG_01_Grayscale'.format(image_index))

        # Step 2.D: Downscale the image if necessary, save the factor
        downscale_factor, gray = downscale_image(gray)

        if DEBUG:
            save_to_results_folder(gray, 'Image-{:03d}_DEBUG_02_Downscale'.format(image_index))

        # Step 2.E: Find the contours
        contours = find_image_contours(gray, use_adaptive_treshold, image_index)

        # Step 2.G: Find the groups of bars
        height, width = gray.shape[:2]
        # Info: max_contour_area is a value computed from the area of the downscaled image
        # used exclude / discard to big contours
        max_contour_area = (width * height) / MIN_SPLIT_BARECODE
        groups = find_groups(contours, max_contour_area)

        # Step 2.H: Extract barcodes
        scale_factor = (1.0 / downscale_factor)
        extract_barcodes(image, groups, scale_factor, image_index)

    # Step 3: Delete the created results folder if empty
    if len(os.listdir(PATH_DIR_RESULTS)) == 0:
        delete_results_folder()

    return True

if __name__ == '__main__':
    import time
    import argparse
    parser = argparse.ArgumentParser(prog=SCRIPT_NAME, description='{} v{}, Detect and Extract 1D Barecode in photographs.'.format(SCRIPT_NAME, VERSION))
    parser.add_argument('-v', '--version', action='version', version='%(prog)s '+ VERSION)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-a', '--adaptive-treshold', action='store_true')
    parser.add_argument('-i', '--images-paths', nargs='+', required=True, type=str, action='extend', help='disk path(s) to the image(s)')

    arguments = parser.parse_args()
    DEBUG = arguments.debug

    start_time = time.time()
    main(arguments.images_paths, use_adaptive_treshold=arguments.adaptive_treshold)
    LOG.info("Execution time: {} seconds.".format((time.time() - start_time)))