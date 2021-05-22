import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL]
x = [i for i in range(10, 65)]
y = [i for i in range(70, 121)]
size = [2, 3]
font_weight = [i for i in range(1, 11)]

os.makedirs('../Data/Data_gen')
for alp in alphabets:
    os.makedirs(f'../Data/Data_gen/{alp}')
    for counter in range(2000):
        font_ = fonts[np.random.randint(len(fonts) - 1)]
        x_ = x[np.random.randint(len(x) - 1)]
        y_ = y[np.random.randint(len(y) - 1)]
        size_ = size[np.random.randint(len(size) - 1)]
        font_weight_ = font_weight[np.random.randint(len(font_weight) - 1)]
        image = np.ones((128, 128))
        cv2.putText(image, alp, (x_, y_), font_,
                    size_, (0, 0, 0), font_weight_)
        image = cv2.convertScaleAbs(image, alpha=(255.0))
        cv2.imwrite(f'../Data/Data_gen/{alp}/{str(counter)}.jpg', image)
    print(f'Completed : {alp}')
