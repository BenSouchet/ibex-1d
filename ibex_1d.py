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

SCRIPT_NAME = 'IBEX 1D : Image 1D Barcode EXtractor'
__version__ = '1.2.0'

RESULT_IMAGE_EXT = '.png'             # Can be any type handled by OpenCV, see documentation for valid values.
DETECTION_IMAGE_MAX_DIM = 1024        # In pixels, if the largest dimension (width or height) of the input image is
#                                       bigger than this value the image will be downscale ONLY for paper detect
#                                       calculations. Smaller value mean faster computation but less accuracy.
MIN_COMPONENT_AREA = DETECTION_IMAGE_MAX_DIM * .05  # Contours below this value will be skipped.
MIN_BARCODE_RECT_RATIO = 4.           # Minimum ratio to consider a rectangle to be part of a barcode
SIMPLIFIED_CONTOUR_MAX_COEF = .15     # Maximum ratio of simplification allowed for the contour point reduction
#                                       (e.g. simplify_contour function)
MIN_SPLIT_BARCODE = 70                # Minium number of columns in a barcode
MIN_BAR_NUMBER = 18                   # Minium number of bars in a group to consider it a barcode
BAR_DETECTION_FACTOR = 4.             # A multiplication factor used for the radius to detect bars of the same barcode
SCALE_DETECTED_BARCODE_FACTOR = 1.04  # A scale value superior to 1.0 to add an offset to detected barcode shape
BARCODE_DEFORMATION_TOLERANCE = 0.01  # Above this value a complex method will be used to compute paper aspect ratio

PATH_DIR_RESULTS = Path.cwd().joinpath('results')
PATH_DIR_CURRENT_RESULTS = Path()
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


def init_logger() -> logging.Logger:
    """Initialize script logger"""
    logger_name = Path(__file__).stem
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=logging.DEBUG)
    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(console_handler)

    return logger


LOG = init_logger()


def delete_folder(path: Path) -> bool:
    """Delete a folder.

    :param path: Path to an existing folder
    :type path: Path
    :return: Success status of the delete operation
    :rtype: bool
    """
    if path and path.exists():
        try:
            shutil.rmtree(path)
        except Exception as e:
            LOG.error('Failed to delete the folder "{}". Reason: {}'.format(path, e))
            return False
        return True
    return False


def create_results_folder() -> bool:
    """Create the folder that is used to store results barcodes extracted.

    :return: Success status of the creation operation
    :rtype: bool
    """
    # Step 1: Create the directory path for the results and set it as a global variable.
    global PATH_DIR_CURRENT_RESULTS
    PATH_DIR_CURRENT_RESULTS = PATH_DIR_RESULTS.joinpath(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Step 2: Creating the directories
    try:
        PATH_DIR_CURRENT_RESULTS.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        LOG.error('Failed to create results folder. Reason: {}'.format(e))
        return False

    return True


def save_to_results_folder(image: NDArray[numpy.uint8], filename: str):
    """Save an image (NumPy array) to the results' folder.

    :param image: NumPy array that represent an image
    :type image: NDArray[numpy.uint8]
    :param filename: The filename (without the extension) of the image to save
    :type filename: str
    """
    cv.imwrite(str(PATH_DIR_CURRENT_RESULTS.joinpath(filename + RESULT_IMAGE_EXT)), image)


def downscale_image(image: NDArray[numpy.uint8]) -> tuple[float, NDArray[numpy.uint8]]:
    """Downscale/resize an image (NumPy array) to ensure max size is smaller that DETECTION_IMAGE_MAX_DIM.

    :param image: NumPy array that represent an image
    :type image: NDArray[numpy.uint8]
    :return: A tuple containing the scale factor (float) and the resized image
    :rtype: tuple[float, NDArray[numpy.uint8]]
    """
    factor = 1.0
    height, width = image.shape[:2]

    # Step 1: If image doesn't need resize do nothing
    if height <= DETECTION_IMAGE_MAX_DIM and width <= DETECTION_IMAGE_MAX_DIM:
        return factor, image

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
    return factor, cv.resize(image, (width, height), interpolation=cv.INTER_AREA)


def split_grayscale(image: NDArray[numpy.uint8]) -> tuple[NDArray[numpy.uint8], NDArray[numpy.uint8]]:
    """Convert multi-channels image to YUV to get the Y channel (luminance component).

    :param image: NumPy array that represent an image
    :type image: NDArray[numpy.uint8]
    :return: A tuple containing the Y channel and the image in YUV format
    :rtype: tuple[NDArray[numpy.uint8], NDArray[numpy.uint8]]
    """
    if len(image.shape) >= 3:
        yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        return yuv[:, :, 0], yuv
    return image, image


def _simplify_contour_compute_weight(contour: NDArray[numpy.float32], index: int) -> float:
    """Compute the weight / importance of a point in a OpenCV contour. The weight is currently the area of the triangle
    formed by the point whose index has been passed, and it's two direct neighbours.

    :param contour: NumPy array that represent a OpenCV contour
    :type contour: NDArray[numpy.float32]
    :param index: Index of the point we want to determine the weight in this contour
    :type index: int
    :return: The weight of the point to which the index refers
    :rtype: float
    """
    p1 = contour[(index - 1) % contour.shape[0]][0]
    p2 = contour[index][0]
    p3 = contour[(index + 1) % contour.shape[0]][0]
    return 0.5 * abs((p1[0] * (p2[1] - p3[1])) + (p2[0] * (p3[1] - p1[1])) + (p3[0] * (p1[1] - p2[1])))


def simplify_contour(contour: NDArray[numpy.float32], nbr_ptr_limit: int = 4) -> NDArray[numpy.float32]:
    """Simplify the contour by reducing the number of points using a naive version of Visvalingam-Whyatt simplification
    algorithm.

    :param contour: NumPy array that represent a OpenCV contour
    :type contour: NDArray[numpy.float32]
    :param nbr_ptr_limit: Number of points that the contour must be composed of after the simplification
    :type nbr_ptr_limit: int
    :return: The simplified contour
    :rtype: NDArray[numpy.float32]
    """
    # points_weights will be used to determine the importance of points,
    # in the Visvalingam-Whyatt algorithm it's the area of the triangle created by a point and his direct neighbours
    points_weights = numpy.zeros(contour.shape[0])

    # Step 1: First pass, computing all points weight
    for index in range(contour.shape[0]):
        points_weights[index] = _simplify_contour_compute_weight(contour, index)

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
        index_pnt_prev = (index_pnt - 1) % contour.shape[0]
        index_pnt_next = index_pnt % contour.shape[0]
        points_weights[index_pnt_prev] = _simplify_contour_compute_weight(contour, index_pnt_prev)
        points_weights[index_pnt_next] = _simplify_contour_compute_weight(contour, index_pnt_next)

    return contour


def are_angles_equal(angle1: float, angle2: float, tolerance: float) -> bool:
    """Check if two angles are equal, minus some tolerance value.

    :param angle1: Angle in radians
    :type angle1: float
    :param angle2: Angle in radians
    :type angle2: float
    :param tolerance: A tolerance value (epsilon), to avoid false negatives
    :type tolerance: float
    :return: Boolean value, True if angle are considered equal, False otherwise.
    :rtype: bool
    """
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


def normalize_vector(vector: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
    """Normalize / converting to unit vector

    :param vector: A vector
    :type vector: NDArray[numpy.float32]
    :return: The normalized vector
    :rtype: NDArray[numpy.float32]
    """
    length = numpy.linalg.norm(vector)
    if length == 0:
        return vector
    return vector / length


def scale_contour_from_centroid(contour: NDArray[numpy.float32], scale: float) -> NDArray[numpy.float32]:
    """Scale an OpenCV contour from its centroid.

    :param contour: NumPy array that represent a OpenCV contour
    :type contour: NDArray[numpy.float32]
    :param scale: Scale value, value in range [0, inf], a negative value should work but the contour will be "flipped"
    :type scale: float
    :return: The scaled contour
    :rtype: NDArray[numpy.float32]
    """
    # Step 1: Determine the centroid of the contour
    moment = cv.moments(contour)
    center_x = int(moment['m10'] / moment['m00'])
    center_y = int(moment['m01'] / moment['m00'])

    # Step 2: move the contour center at 0,0
    contour_normalized = contour - [center_x, center_y]

    # Step 3: Scale
    contour = contour_normalized * scale

    # Step 4: Move back the contour to it position
    contour += [float(center_x), float(center_y)]

    return contour


def _get_corners_from_contour(contour: NDArray[numpy.float32]) -> list[tuple[float, float]]:
    """Contour should be simplified to 4 points, in this function the goal is to ensure that these points are
    properly ordered, first point should be the one closest to the top-left corner of the photograph and the remaining
    points should be clockwise ordered.

    :param contour: NumPy array that represent a OpenCV contour
    :type contour: NDArray[numpy.float32]
    :return: List of 4 2D points, clockwise ordered, that represent a quadrangle
    :rtype: list[tuple[float, float]]
    """
    # Remove useless dimensions
    contour = numpy.squeeze(contour)

    # We need first to ensure a clockwise orientation for the contour
    corners = None

    # Step 1: Find top left point, using distance to top left of the picture
    dist_list = [[numpy.linalg.norm(point), index] for index, point in enumerate(contour)]
    dist_list = sorted(dist_list, key=lambda x: x[0])

    index_pnt_tl = dist_list[0][1]

    # Step 2: Find the others points order. Since the contour has been retrieved via
    #         cv.findContours it's either sorted in clockwise or counter-clockwise,
    count_points = 4  # We know at this point that the contour as only 4 points, no more, no less
    index_pnt_prev = (index_pnt_tl - 1) % count_points
    index_pnt_next = (index_pnt_tl + 1) % count_points
    index_pnt_last = (index_pnt_tl + 2) % count_points

    # Step 2.B: Comparing x-axis values of the neighbours of the top left point find out if the
    #           contour has been sorted in clockwise or counter-clockwise
    if contour[index_pnt_prev][0] > contour[index_pnt_next][0]:
        # Counter-clockwise
        corners = [contour[index_pnt_tl], contour[index_pnt_prev],
                   contour[index_pnt_last], contour[index_pnt_next]]
    else:
        # Clockwise
        corners = [contour[index_pnt_tl], contour[index_pnt_next],
                   contour[index_pnt_last], contour[index_pnt_prev]]

    return corners


class BarCandidate:
    points: list[tuple[float, float]]
    dimensions: tuple[float, float]
    centroid: tuple[float, float]
    area: float
    angle: float
    detection_radius: float

    def compute_angle(self):
        """Compute (and store) the radians angle of the bar with an up vector"""
        v0 = numpy.array((0, 1))
        v1 = normalize_vector(numpy.array(self.points[3]) - numpy.array(self.points[0]))
        self.angle = numpy.arctan2(numpy.linalg.det([v0, v1]), numpy.dot(v0, v1))

    def __init__(self, contour: NDArray[numpy.float32], area: float, dimensions: tuple[float, float]):
        # Points
        self.points = _get_corners_from_contour(contour)
        # Dimensions
        self.dimensions = dimensions
        # Centroid
        m = cv.moments(contour)
        self.centroid = (m['m10'] / m['m00'], m['m01'] / m['m00'])
        # Area
        self.area = area
        # Angle (in radians)
        self.compute_angle()
        # Detection Radius
        self.detection_radius = min(self.dimensions[0], self.dimensions[1]) * BAR_DETECTION_FACTOR


def compute_aspect_ratio(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32]) -> float:
    """Compute aspect ratio of a quadrangle in a (not cropped) photograph.

    :param image: NumPy array that represent an image
    :type image: NDArray[numpy.uint8]
    :param corners: List of 4 2D points, clockwise ordered, that represent a quadrangle
    :type corners: NDArray[numpy.float32]
    :return: Aspect ratio of the quadrangle in the image
    :rtype: float
    """
    # Based on :
    # - https://www.microsoft.com/en-us/research/publication/2016/11/Digital-Signal-Processing.pdf
    # - http://research.microsoft.com/en-us/um/people/zhang/papers/tr03-39.pdf
    # - https://andrewkay.name/blog/post/aspect-ratio-of-a-rectangle-in-perspective/

    # Step 1: Get image center, will be used as origin
    h, w = image.shape[:2]
    origin = (w * .5, h * .5)

    # Step 2: Homogeneous points coords from image origin
    # /!\ CAREFUL : points need to be in zigzag order (A, B, D, C)
    p1 = numpy.array([*(corners[0] - origin), 1.])
    p2 = numpy.array([*(corners[1] - origin), 1.])
    p3 = numpy.array([*(corners[3] - origin), 1.])
    p4 = numpy.array([*(corners[2] - origin), 1.])

    # Step 3: Zhengyou Zhang p.10 : equations (11) & (12)
    k2 = numpy.dot(numpy.cross(p1, p4), p3) / numpy.dot(numpy.cross(p2, p4), p3)
    k3 = numpy.dot(numpy.cross(p1, p4), p2) / numpy.dot(numpy.cross(p3, p4), p2)

    # Step 4: Compute the focal length
    f = 0.
    f_sq = -((k3 * p3[1] - p1[1]) * (k2 * p2[1] - p1[1]) +
             (k3 * p3[0] - p1[0]) * (k2 * p2[0] - p1[0])) / ((k3 - 1) * (k2 - 1))
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

    return len_v / len_u


def compute_barcode_size(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32]) -> tuple[int, int]:
    """Compute a barcode size from its shape.

    :param image: NumPy array that represent an image
    :type image: NDArray[numpy.uint8]
    :param corners: List of 4 2D points, clockwise ordered, that represent a quadrangle
    :type corners: NDArray[numpy.float32]
    :return: Tuple of two values, width and height
    :rtype: tuple[int, int]
    """
    # Vectors of the side of the contour (clockwise)
    side_top_vec = corners[1] - corners[0]
    side_rgh_vec = corners[2] - corners[1]
    side_btm_vec = corners[2] - corners[3]
    side_lft_vec = corners[3] - corners[0]

    # Step 1: Compute average width & height of the paper sheet
    paper_avg_width = 0.5 * (numpy.linalg.norm(side_top_vec) + numpy.linalg.norm(side_btm_vec))
    paper_avg_height = 0.5 * (numpy.linalg.norm(side_lft_vec) + numpy.linalg.norm(side_rgh_vec))

    # Step 2: If deformation is negligible avoid computation and return the average dimensions
    #         Checking if the opposite sides are parallel
    if math.isclose((side_top_vec[0] * side_btm_vec[1]), (side_top_vec[1] * side_btm_vec[0]),
                    abs_tol=BARCODE_DEFORMATION_TOLERANCE) and \
            math.isclose((side_lft_vec[0] * side_rgh_vec[1]), (side_lft_vec[1] * side_rgh_vec[0]),
                         abs_tol=BARCODE_DEFORMATION_TOLERANCE):
        return round(paper_avg_width), round(paper_avg_height)

    # Step 3: Compute aspect ratio
    aspect_ratio = compute_aspect_ratio(image, corners)

    if aspect_ratio == 0.:
        # The ratio could not be computed, use a fallback
        rect = cv.minAreaRect(corners)
        return rect.size.width, rect.size.height

    return round(paper_avg_width), round(paper_avg_width * aspect_ratio)


def extract_barcode(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32]) -> NDArray[numpy.uint8]:
    """Extract a barcode from an image based on is deformed quadrangle shape (4 points) and return it as a new image.
    The barcode is unwrapped/straighten by first computing the perspective matrix.

    :param image: NumPy array that represent an image
    :type image: NDArray[numpy.uint8]
    :param corners: List of 4 2D points, clockwise ordered, that represent a quadrangle
    :type corners: NDArray[numpy.float32]
    :return: NumPy array that represent the barcode image
    :rtype: NDArray[numpy.uint8]
    """
    # Step 1: Compute size (width & height) of the barcode
    (width, height) = compute_barcode_size(image, corners)

    # Step 2: Create the destination image size array
    dim_dest_image = numpy.array([[0., 0.], [(width - 1.), 0.], [(width - 1.), (height - 1.)], [0., (height - 1.)]])

    # Step 3: Compute the perspective deformation matrix
    #         /!\ inputs need to be numpy array in float32
    m = cv.getPerspectiveTransform(corners.astype(numpy.float32), dim_dest_image.astype(numpy.float32))

    # Step 4: Extract and unwrap/straighten the barcode
    barcode = cv.warpPerspective(image, m, (int(width), int(height)), borderMode=cv.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

    return barcode


def _find_candidates(image: NDArray[numpy.uint8], use_adaptive_threshold: bool, image_index: int) -> list[BarCandidate]:
    """Find bar candidates of the barcode(s)

    :param image: NumPy array that represent an image
    :type image: NDArray[numpy.uint8]
    :param use_adaptive_threshold: Whether the adaptive threshold should be use instead of a classic OTSU threshold.
    :type use_adaptive_threshold: bool
    :param image_index: Index of the current processed image, used for result(s) filename(s).
    :type image_index: int
    :return: A list of bar candidates
    :rtype: list[BarCandidate]
    """
    height, width = image.shape[:2]

    # Step 1: Apply threshold to be able to detect barcode lines
    if use_adaptive_threshold:
        # Block size is 1/10 of the smallest of the two dimensions rounded to the nearest odd number
        block_size = int(math.ceil(min(width, height) * .05) * 2 + 1)
        c_const = round(block_size * .1)  # 1/10 of block size
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, block_size, c_const)
    else:
        image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    if DEBUG:
        save_to_results_folder(image, 'Image-{:03d}_DEBUG_03_Threshold'.format(image_index))

    # Step 2: Create candidates list

    max_component_area = (width * height) / MIN_SPLIT_BARCODE  # Used to discard too big components

    # Step 2.A: Compute Connected Components
    candidates = []
    (count_labels, labels, stats, _) = cv.connectedComponentsWithStatsWithAlgorithm(image, connectivity=8,
                                                                                    ltype=cv.CV_32S,
                                                                                    ccltype=cv.CCL_GRANA)
    for index in range(1, count_labels):
        # Step 2.B: Filter too small or too big components
        area = stats[index, cv.CC_STAT_AREA]
        if area < MIN_COMPONENT_AREA or area > max_component_area:
            continue

        # Step 2.C: Create a custom ROI "image", containing only the current component
        x = stats[index, cv.CC_STAT_LEFT]
        y = stats[index, cv.CC_STAT_TOP]
        w = stats[index, cv.CC_STAT_WIDTH]
        h = stats[index, cv.CC_STAT_HEIGHT]

        roi_label = labels[y:y + h, x:x + w]
        roi = (roi_label == index).astype("uint8") * 255

        # Step 2.D: Find contours inside this ROI
        roi_contours, _ = cv.findContours(roi, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE, offset=(x, y))

        # Step 2.E: Merge all these contours and get the min rect
        points = numpy.empty(shape=(0, 1, 2), dtype=numpy.int32)

        for roi_contour in roi_contours:
            points = numpy.append(points, roi_contour, 0)

        rect = cv.minAreaRect(points)

        # Step 2.F: Filter the components that doesn't look like bars
        (width, height) = rect[1]
        if (max(width, height) / max(min(width, height), 1.)) < MIN_BARCODE_RECT_RATIO:
            continue

        candidates.append(BarCandidate(numpy.squeeze(cv.boxPoints(rect)), area, rect[1]))

    return candidates


def _find_groups(candidates: list[BarCandidate]) -> list[list[BarCandidate]]:
    """Try regrouping bar candidates, these groups represent barcodes.

    :param candidates: List of bar candidates that's need to be regrouped
    :type candidates: list[BarCandidate]
    :return: A list of group of bar candidates, each group should represent a barcode
    :rtype: list[list[BarCandidate]]
    """
    # Step 1: Group linked bars, approx similar angles + close to each others
    groups = []
    while len(candidates):
        group = [candidates.pop()]
        for current in group:
            for index_candidate, candidate in reversed(list(enumerate(candidates))):
                if are_angles_equal(current.angle, candidate.angle, tolerance=0.07) and \
                                    numpy.linalg.norm(numpy.subtract(candidate.centroid, current.centroid)) <= max(
                                    current.detection_radius, candidate.detection_radius):
                    group.append(candidates.pop(index_candidate))
        groups.append(group)

    return groups


def _is_rotation_needed(corners: NDArray[numpy.float32]) -> bool:
    """Check if a rotation is necessary/needed for the input shape (quadrangle).

    :param corners: List of 4 2D points, clockwise ordered, that represent a quadrangle
    :type corners: NDArray[numpy.float32]
    :return: Whether a rotation is needed for the quadrangle
    :rtype: bool
    """
    # Step 1: Compute vectors of edge 1 and 2 of the shape
    side_top_vec = corners[1] - corners[0]
    side_rgh_vec = corners[2] - corners[1]

    # Step 2: Determine if the result need a 90 degrees rotation
    #         we can do a simple test because we have ordered the corners in clockwise
    #         and first point is the closest to top left corner of image.
    if numpy.linalg.norm(side_top_vec) < numpy.linalg.norm(side_rgh_vec):
        # First edge smaller than second, shape need the rotation
        return True
    return False


def _extract_barcodes_internal(image: NDArray[numpy.uint8], groups: list[list[BarCandidate]], scale_factor: float,
                               image_index: int):
    """Extract barcode(s) in an image.

    :param image: NumPy array that represent an image
    :type image: NDArray[numpy.uint8]
    :param groups: A list of group of bar candidates, each group should represent a barcode
    :type groups: list[list[BarCandidate]]
    :param image_index: Index of the current processed image, used for result(s) filename(s).
    :type image_index: int
    """
    barcode_index = 1

    for group in groups:
        if len(group) < MIN_BAR_NUMBER:
            # Not enough bars to consider this group valid
            continue

        # Step 1: Get the enclosing convex shape
        # Step 1.A: Append the points of all the bars of the groupAdd points of the current
        points = []
        for bar in group:
            points.extend(bar.points)

        # Step 1.B: Reshape to have same format as a contour to call convexhull OpenCV method
        temp_contour = numpy.array(points).reshape((-1, 1, 2)).astype(numpy.int32)
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

        # Step 2: Scale up a bit the shape to add space
        hull = scale_contour_from_centroid(hull, SCALE_DETECTED_BARCODE_FACTOR)

        # Step 3: Very important /!\ Apply the scale factor to scale up the contours to the correct size
        hull = hull * scale_factor
        corners = numpy.array(_get_corners_from_contour(hull))

        if DEBUG:
            debug_image = image.copy()
            cv.drawContours(debug_image, [corners.astype(numpy.int32)], 0, (0, 0, 255), 1, cv.LINE_AA)
            save_to_results_folder(debug_image, 'Image-{:03d}_DEBUG_04_barcode{:03d}'.format(image_index,
                                                                                             barcode_index))

        # Step 4: Extract / unwrap the barcode
        barcode_image = extract_barcode(image, corners)

        # Step 5: Ensure the barcode is properly rotated (horizontal)
        if _is_rotation_needed(corners):
            barcode_image = cv.rotate(barcode_image, cv.ROTATE_90_CLOCKWISE)

        # Step 6: Save to folder
        filename = "Image-{:03d}_Barcode-{:03d}".format(image_index, barcode_index)
        save_to_results_folder(barcode_image, filename)

        barcode_index += 1


def extract_barcodes(images_paths: list[str], use_adaptive_threshold: bool = False) -> bool:
    """Extract barcodes from photographs, best/optimal results if photographs hasn't been cropped.

    :param images_paths: List of paths to the images, in each image there should be one or more barcode(s) to extract.
    :type images_paths: list[str]
    :param use_adaptive_threshold: Whether the adaptive threshold should be use instead of a classic OTSU threshold.
    :type use_adaptive_threshold: bool
    :return: The execution result status, False means error(s)/issue(s) has occurred.
    :rtype: bool
    """
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

        # Step 2.E: Find the bars candidates
        candidates = _find_candidates(gray, use_adaptive_threshold, image_index)
        if not candidates:
            continue

        # Step 2.G: Find the groups of bars
        groups = _find_groups(candidates)

        # Step 2.H: Extract barcodes
        scale_factor = (1.0 / downscale_factor)
        _extract_barcodes_internal(image, groups, scale_factor, image_index)

    # Step 3: Delete the created results folder(s) if empty
    if len(os.listdir(PATH_DIR_CURRENT_RESULTS)) == 0:
        success = delete_folder(PATH_DIR_CURRENT_RESULTS)
        if success and len(os.listdir(PATH_DIR_RESULTS)) == 0:
            delete_folder(PATH_DIR_RESULTS)

    return True


def main() -> bool:
    """Entry point when the package is executed as a script from a command line.

    :return: The execution result status, False means error(s)/issue(s) has occurred.
    :rtype: bool
    """
    import time
    import argparse

    parser = argparse.ArgumentParser(prog=SCRIPT_NAME,
                                     description='{} v{}, Detect and Extract 1D Barcodes in Photographs.'.format(
                                         SCRIPT_NAME, __version__))
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-a', '--adaptive-threshold', action='store_true')
    parser.add_argument('-i', '--images-paths', nargs='+', required=True, type=str, action='extend',
                        help='disk path(s) to the image(s)')

    arguments = parser.parse_args()
    global DEBUG
    DEBUG = arguments.debug

    start_time = time.time()
    return_value = extract_barcodes(arguments.images_paths, use_adaptive_threshold=arguments.adaptive_threshold)

    if DEBUG:
        LOG.info("Execution time: {} seconds.".format((time.time() - start_time)))

    return return_value


if __name__ == '__main__':
    main()
