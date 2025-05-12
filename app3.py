import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def debug_model_inputs(cells, model):
    fig, axes = plt.subplots(9, 9, figsize=(12, 12))
    plt.tight_layout(pad=0.8)

    for idx, cell in enumerate(cells):
        row, col = divmod(idx, 9)
        ax = axes[row][col]

        prepared = prepare_cell_for_model(cell, debug=True, cell_index=idx)
        if prepared is None:
            ax.imshow(cell, cmap="gray")
            ax.set_title("0", fontsize=8, color="blue")
        else:
            prediction = model.predict(prepared, verbose=0)
            digit = np.argmax(prediction)
            ax.imshow(prepared.reshape(28, 28), cmap="gray")
            ax.set_title(str(digit), fontsize=8, color="green")

        ax.axis("off")

    plt.suptitle("Model Inputs and Predictions", fontsize=16)
    plt.show()

def load_and_warp(path):
    # img = cv2.imread(path)
    gray = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
    if len(approx) != 4:
        raise Exception("Sudoku grid not found.")

    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)]  # bottom-left
    ], dtype="float32")

    side = max([
        np.linalg.norm(ordered[0] - ordered[1]),
        np.linalg.norm(ordered[2] - ordered[3]),
        np.linalg.norm(ordered[0] - ordered[3]),
        np.linalg.norm(ordered[1] - ordered[2]),
    ])
    side = int(side)

    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(gray, M, (side, side))

    return warped

def binarize_and_denoise(warped):
    blur = cv2.GaussianBlur(warped, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned


def split_into_cells(thresh_img):
    cells = []
    side = thresh_img.shape[0]
    cell_size = side // 9

    for i in range(9):
        for j in range(9):
            y1 = i * cell_size
            y2 = (i + 1) * cell_size
            x1 = j * cell_size
            x2 = (j + 1) * cell_size


            margin = 6
            top = margin if i != 0 else 4
            bottom = margin if i != 8 else 4
            left = margin if j != 0 else 4
            right = margin if j != 8 else 4

            cell = thresh_img[y1+top:y2-bottom, x1+left:x2-right]
            cells.append(cell)

    return cells


def prepare_cell_for_model(cell, debug=False, cell_index=None):
    cell = cv2.resize(cell, (64, 64), interpolation=cv2.INTER_AREA)
    white_pixels = cv2.countNonZero(cell)
    if white_pixels < (cell.shape[0] * cell.shape[1] * 0.03):
        if debug: print(f"[{cell_index}] Skipped: too empty")
        return None

    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug: print(f"[{cell_index}] Skipped: no contours")
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    if w < 2 or h < 5 or w * h < 40:
        if debug: print(f"[{cell_index}] Skipped: too small (w={w}, h={h}, area={w*h})")
        return None


    if x <= 0 or y <= 0 or x + w >= 63 or y + h >= 63:
        if debug: print(f"[{cell_index}] Skipped: touches border")
        return None

    digit = cell[y:y+h, x:x+w]

    if w < 15 or h < 15:
        digit = cv2.dilate(digit, np.ones((2, 2), np.uint8), iterations=1)

    square = np.zeros((28, 28), dtype=np.uint8)
    digit = cv2.resize(digit, (18, 18), interpolation=cv2.INTER_AREA)
    square[5:23, 5:23] = digit

    square = square.astype("float32") / 255.0
    return square.reshape(1, 28, 28, 1)



def predict_board(cells, model):
    board = []
    for cell in cells:
        prepared = prepare_cell_for_model(cell)
        if prepared is None:
            board.append(0)
        else:
            prediction = model.predict(prepared, verbose=0) 
            digit = np.argmax(prediction)
            # if prediction[0][digit] >= 0.6:
            #     board.append(digit)
            # else:
            #     board.append(0)
            board.append(digit)
    return np.array(board).reshape((9, 9))

model = load_model("v2_printed_digits_cnn.h5")

def image_to_sudoku(path):
    warped = load_and_warp(path)
    cleaned = binarize_and_denoise(warped)
    # imgage = plt.imshow(cleaned)
    # plt.show()
    cells = split_into_cells(cleaned)




    debug_model_inputs(cells, model)
    board = predict_board(cells, model)


    print(board)
    return board



