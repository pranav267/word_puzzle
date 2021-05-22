import sys
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import streamlit as st


def img_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            edges = cv2.approxPolyDP(i, 0.02*perimeter, True)
            if area > max_area and len(edges) == 4:
                biggest = edges
                max_area = area
    return biggest, max_area


def reorder(edges):
    edges = edges.reshape((4, 2))
    edges_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = edges.sum(1)
    edges_new[0] = edges[np.argmin(add)]
    edges_new[3] = edges[np.argmax(add)]
    diff = np.diff(edges, axis=1)
    edges_new[1] = edges[np.argmin(diff)]
    edges_new[2] = edges[np.argmax(diff)]
    return edges_new


def split_boxes(img, nrows, ncols):
    rows = np.vsplit(img, nrows)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, ncols)
        row_boxes = []
        for box in cols:
            row_boxes.append(box)
        boxes.append(row_boxes)
    return boxes


def preprocess_test_image(img):
    IMG_SIZE = 32
    # img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_new = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_new = img_new / 255.
    return img_new.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def load_models():
    model_names = ['Model(97.09)', 'Model(97.64)',
                   'Model(98.79)', 'Model(98.94)',
                   'Model(95.95)']
    # model_names = ['Model(97.09)', 'Model(97.64)',
    #                'Model(98.79)', 'Model(98.94)',
    #                'Model(95.95)', 'Model(93.43)',
    #                'Model(93.48)']
    models = []
    for m in model_names:
        json_file = open(f'../Models/{m}.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(f'../Models/{m}.h5')
        models.append(model)
    # print('Models Loaded From Disk!')
    return models


def get_prediction(img, models):
    classes = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    preds = [model.predict([img])[0] for model in models]
    preds = np.array(preds)
    summed = np.sum(preds, axis=0)
    pred = classes[int(np.argmax(summed))]
    return pred


def predict_alphabet(img, models):
    img = preprocess_test_image(img)
    pred = get_prediction(img, models)
    return pred


def get_grid(boxes, nrows, ncols):
    matrix = []
    counter = 1
    models = load_models()
    elements = nrows * ncols
    st.markdown("""
    <style>
    .stProgress .st-bo {
        background-color: red;
    }
    </style>
    """, unsafe_allow_html=True)
    my_bar = st.progress(0)
    for r in boxes:
        row_matrix = []
        for c in r:
            perct = np.round((counter / elements) * 100, 2)
            perct_ = int((counter / elements) * 100)
            alpha = predict_alphabet(c, models)
            row_matrix.append(alpha)
            prect_s = f'Processing : {perct}'
            sys.stdout.write('\r'+str(prect_s)+'%')
            sys.stdout.flush()
            counter += 1
            my_bar.progress(perct_)
        matrix.append(row_matrix)
    return matrix


def search_word_helper(grid, row, col, word, R, C):
    dir = [[-1, 0], [1, 0], [1, 1], [1, -1],
           [-1, -1], [-1, 1], [0, 1], [0, -1]]
    if grid[row][col] != word[0]:
        return False
    for x, y in dir:
        rd, cd = row + x, col + y
        flag = True
        for k in range(1, len(word)):
            if (0 <= rd < R and 0 <= cd < C and word[k] == grid[rd][cd]):
                rd += x
                cd += y
            else:
                flag = False
                break
        if flag:
            return True
    return False


def search_word_main(grid, word):
    coordinates = []
    R = len(grid)
    C = len(grid[0])
    for row in range(R):
        for col in range(C):
            if search_word_helper(grid, row, col, word, R, C):
                coordinates.append(row)
                coordinates.append(col)
                return coordinates
    coordinates.append(-1)
    return coordinates


def solve_puzzle(word_matrix, word_list):
    word_coordinates = {}
    words = [i.upper() for i in word_list]
    for word in words:
        coordinates = search_word_main(word_matrix, word)
        if coordinates[0] == -1:
            word_coordinates[word] = [-1, -1]
        else:
            word_coordinates[word] = coordinates
    return word_coordinates


def get_dir_words(word, grid, x, y, nrows, ncols):
    word1 = ''
    word2 = ''
    word3 = ''
    word4 = ''
    word5 = ''
    word6 = ''
    word7 = ''
    word8 = ''
    for i in range(len(word)):
        if x + i < nrows:
            word3 += grid[x + i][y]
        if x - i >= 0:
            word7 += grid[x - i][y]
        if y + i < ncols:
            word1 += grid[x][y + i]
        if y - i >= 0:
            word5 += grid[x][y - i]
        if x + i < nrows and y + i < ncols:
            word2 += grid[x + i][y + i]
        if x + i < nrows and y - i >= 0:
            word4 += grid[x + i][y - i]
        if x - i >= 0 and y - i >= 0:
            word6 += grid[x - i][y - i]
        if x - i >= 0 and y + i < ncols:
            word8 += grid[x - i][y + i]
    if word1 == word:
        return 1
    if word2 == word:
        return 2
    if word3 == word:
        return 3
    if word4 == word:
        return 4
    if word5 == word:
        return 5
    if word6 == word:
        return 6
    if word7 == word:
        return 7
    if word8 == word:
        return 8


def plot_lines(black_boxes, word_positions):
    for word in word_positions:
        l = len(word)
        x = word_positions[word][0]
        y = word_positions[word][1]
        d = word_positions[word][2]
        h = black_boxes[x][y].shape[0]
        w = black_boxes[x][y].shape[1]
        if d == 1:
            for i in range(len(word)):
                cv2.line(black_boxes[x][y + i], (0, int(h/2)),
                         (w, int(h/2)), (255, 0, 0), 90)
        elif d == 2:
            for i in range(len(word)):
                cv2.line(black_boxes[x + i][y + i],
                         (0, 0), (w, h), (255, 0, 0), 90)
        elif d == 3:
            for i in range(len(word)):
                cv2.line(black_boxes[x + i][y], (int(w/2), 0),
                         (int(w/2), h), (255, 0, 0), 90)
        elif d == 4:
            for i in range(len(word)):
                cv2.line(black_boxes[x + i][y - i],
                         (0, h), (w, 0), (255, 0, 0), 90)
        if d == 5:
            for i in range(len(word)):
                cv2.line(black_boxes[x][y - i], (0, int(h/2)),
                         (w, int(h/2)), (255, 0, 0), 90)
        elif d == 6:
            for i in range(len(word)):
                cv2.line(black_boxes[x - i][y - i],
                         (0, 0), (w, h), (255, 0, 0), 90)
        elif d == 7:
            for i in range(len(word)):
                cv2.line(black_boxes[x - i][y], (int(w/2), 0),
                         (int(w/2), h), (255, 0, 0), 90)
        elif d == 8:
            for i in range(len(word)):
                cv2.line(black_boxes[x - i][y + i],
                         (0, h), (w, 0), (255, 0, 0), 90)
    return black_boxes


def get_word_boxes(word_positions, word_matrix, nrows, ncols, black_boxes):
    # print('\n\n')
    print('WORD POSITIONS AND DIRECTIONS')
    new_word_positions = {}
    directions = []
    for word in word_positions:
        word_length = len(word)
        posx = word_positions[word][0]
        posy = word_positions[word][1]
        if posx == -1:
            directions.append(-1)
            continue
        word_directions = get_dir_words(
            word, word_matrix, posx, posy, nrows, ncols)
        print(
            f'{word} is found at ({str(posx)},{str(posy)}) in the direction {str(word_directions)}')
        new_word_positions[word] = [posx, posy, word_directions]
    drawn_boxes = plot_lines(black_boxes, new_word_positions)
    image_mask_temp = []
    for i in drawn_boxes:
        temp_boxx = np.hstack(i)
        image_mask_temp.append(temp_boxx)
    image_mask = np.vstack(image_mask_temp)
    return image_mask


def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(
                    imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver


def word_puzzle(image_path, number_of_rows, number_of_columns, words_list):
    img_path = image_path
    nrows = number_of_rows
    ncols = number_of_columns
    words = words_list
    h = nrows * ncols * 10
    w = nrows * ncols * 10
    img = cv2.imread(img_path)
    img = cv2.resize(img, (w, h))
    imgB = np.zeros((h, w, 3), np.uint8)
    img_pp = img_preprocess(img)
    contours, hierarchy = cv2.findContours(
        img_pp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest, max_area = biggest_contour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warped_c = cv2.warpPerspective(img, matrix, (w, h))
        img_warped_c = cv2.cvtColor(img_warped_c, cv2.COLOR_BGR2GRAY)
    boxes = split_boxes(img_warped_c, nrows, ncols)
    black_box = imgB.copy()
    black_boxes = split_boxes(black_box, nrows, ncols)
    word_matrix = get_grid(boxes, nrows, ncols)
    word_positions = solve_puzzle(word_matrix, words)
    image_mask = get_word_boxes(
        word_positions, word_matrix, nrows, ncols, black_boxes)
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    inv_img_warped_c = img.copy()
    inv_img_warped_c = cv2.warpPerspective(image_mask, matrix, (w, h))
    inv_perspective = cv2.addWeighted(inv_img_warped_c, 0.9, img, 0.7, 1)
    op_img = inv_perspective.copy()
    return op_img
