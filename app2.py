import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your pre-trained digit recognition model
model = load_model("v2_printed_digits_cnn.h5")

def debug_cell_predictions(cells, model):
    fig, axes = plt.subplots(9, 9, figsize=(10, 10))
    plt.tight_layout(pad=0.5)

    for idx, cell in enumerate(cells):
        row, col = divmod(idx, 9)
        ax = axes[row][col]

        # Process the cell
        processed = preprocess_cell(cell)

        if processed is None:
            ax.imshow(cell, cmap='gray')
            ax.set_title("0", fontsize=8, color='blue')  # Blank cell
        else:
            prediction = model.predict(processed, verbose=0)
            digit = np.argmax(prediction)
            ax.imshow(processed.reshape(28, 28), cmap='gray')
            ax.set_title(str(digit), fontsize=8, color='green')

        ax.axis('off')

    plt.show()

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(path, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Use Otsu's binarization for full black/white contrast
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return img, thresh


def find_biggest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Only consider quadrilateral shapes
        if len(approx) == 4 and area > max_area:
            biggest = approx
            max_area = area

    return biggest


def reorder_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left

    return rect


def warp_perspective(img, contour):
    if contour is None or len(contour) != 4:
        raise Exception("Could not find valid Sudoku contour.")

    points = reorder_points(contour)
    side = max([
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]),
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    ])
    side = int(side)

    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(img, M, (side, side))
    return warped


def extract_cells(warped):
    cells = []
    h, w = warped.shape
    cell_h, cell_w = h // 9, w // 9

    for i in range(9):
        for j in range(9):
            y1 = i * cell_h + 6
            y2 = (i + 1) * cell_h - 6  # Crop less on bottom
            x1 = j * cell_w + 4
            x2 = (j + 1) * cell_w - 6  # Crop more on right

            cell = warped[y1:y2, x1:x2]
            cells.append(cell)
    return cells



def preprocess_cell(cell):
    cell = cv2.GaussianBlur(cell, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    h, w = thresh.shape
    white_pixels = cv2.countNonZero(thresh)
    if white_pixels < (h * w * 0.02):
        return None

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)


    if w * h < 100:
        return None

    digit = thresh[y:y+h, x:x+w]


    padded = np.zeros((28, 28), dtype=np.uint8)
    digit = cv2.resize(digit, (18, 18))
    padded[5:23, 5:23] = digit

    padded = padded.astype("float32") / 255.0
    return padded.reshape(1, 28, 28, 1)


def recognize_digits(cells):
    board = []
    for cell in cells:
        processed = preprocess_cell(cell)
        if processed is None:
            board.append(0)
        else:
            prediction = model.predict(processed)
            digit = np.argmax(prediction)
            board.append(digit)
    return np.array(board).reshape((9, 9))

def image_to_sudoku(path):
    img, thresh = preprocess_image(path)
    contour = find_biggest_contour(thresh)
    warped = warp_perspective(img, contour)
    cells = extract_cells(warped)
    debug_cell_predictions(cells, model)

    board = recognize_digits(cells)
    return board

# # Example usage
# sudoku_array = image_to_sudoku("sudoku2.png")
# print(sudoku_array)
