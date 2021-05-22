''' IMPORTS & CONSTANTS '''
from utils import *

img_path = input('ENTER IMAGE PATH : ')
nrows = int(input('ENTER NUMBER OF ROWS : '))
ncols = int(input('ENTER NUMBER OF COLUMNS : '))
number_of_words = int(input('HOW MANY WORDS DO YOU WANT TO SEARCH? : '))
words = []
for i in range(number_of_words):
    s = input(f'ENTER WORD {str(i + 1)} : ')
    words.append(s)
h = nrows * ncols * 10
w = nrows * ncols * 10

''' PREPARE IMAGE '''
img = cv2.imread(img_path)
img = cv2.resize(img, (w, h))
imgB = np.zeros((h, w, 3), np.uint8)
img_pp = img_preprocess(img)

''' FINDING CONTOURS '''
imgC = img.copy()
imgBC = img.copy()
contours, hierarchy = cv2.findContours(
    img_pp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgC, contours, -1, (0, 0, 0), 3)

''' BIGGEST CONTOUR '''
biggest, max_area = biggest_contour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBC, biggest, -1, (0, 0, 0), 20)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped_c = cv2.warpPerspective(img, matrix, (w, h))
    img_warped_c = cv2.cvtColor(img_warped_c, cv2.COLOR_BGR2GRAY)
    imd_detected_alphabets = imgB.copy()

''' SPLIT IMAGE INTO SQUARES OF ALPHABETS '''
boxes = split_boxes(img_warped_c, nrows, ncols)
black_box = imgB.copy()
black_boxes = split_boxes(black_box, nrows, ncols)
box_sample = boxes[0][0]
box_sample = cv2.resize(box_sample, (h, w))

''' RECOGNIZE ALPHABET AND FORM MATRIX '''
word_matrix = get_grid(boxes, nrows, ncols)

print('\n\n')
print('WORD GRID\n')
for r in word_matrix:
    row_string = ' '
    for c in r:
        row_string += c + ' '
    print(row_string)

# for r in range(len(boxes)):
#     for c in range(len(boxes[r])):
#         print(word_matrix[r][c])
#         temp = cv2.resize(boxes[r][c], (64, 64))
#         cv2.imshow('Predictions', temp)
#         key = cv2.waitKey(1000)
#         if key == 27:
#             cv2.destroyAllWindows()
#             break

sample_output = imgB.copy()
cv2.putText(sample_output, word_matrix[0][0], (150, 400),
            cv2.FONT_HERSHEY_SIMPLEX, 15, (255, 255, 255), 18)

''' SOLVE PUZZLE '''
word_positions = solve_puzzle(word_matrix, words)
image_mask = get_word_boxes(
    word_positions, word_matrix, nrows, ncols, black_boxes)

''' INVERSE WARP & OVERLAY IMAGE '''
pts2 = np.float32(biggest)
pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
inv_img_warped_c = img.copy()
inv_img_warped_c = cv2.warpPerspective(image_mask, matrix, (w, h))
inv_perspective = cv2.addWeighted(inv_img_warped_c, 1, img, 0.6, 1)

''' DISPLAY IMAGES '''
# imgArr = ([img, img_pp, imgC, imgBC, img_warped_c], [
#     box_sample, sample_output, image_mask,
#     inv_img_warped_c, inv_perspective])
# stackedImg = stackImages(imgArr, 0.5)
# cv2.imshow('Process', stackedImg)

ip_img = img.copy()
op_img = inv_perspective.copy()
# ip_img = cv2.resize(ip_img, (350, 350))
# cv2.putText(ip_img, 'INPUT IMAGE', (60, 340),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
# op_img = cv2.resize(op_img, (350, 350))
# cv2.putText(op_img, 'OUTPUT IMAGE', (60, 340),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
# image_output = np.hstack([ip_img, op_img])
# cv2.imshow('Output', image_output)
# cv2.waitKey(0)
