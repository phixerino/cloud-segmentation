import numpy as np
import cv2
import os

''' 
check for every pixel if cloud_pixel == (not clear_pixel and not cloud_shadow_pixel)
'''

show = True

dir_path = '/data/jk/sentinel/masks/'

for i, path in enumerate(os.listdir(dir_path)):
    file_path = os.path.join(dir_path, path)
    if os.path.isfile(file_path):
        print(file_path)
        img = np.zeros((1022,1022), dtype=np.uint8)
        img_bool = np.load(file_path)
        img[np.where(img_bool[:, :, 0] == True)] = 255
        img[np.where(img_bool[:, :, 1] == True)] = 0
        img[np.where(img_bool[:, :, 2] == True)] = 0
        img1 = np.where(img_bool[:, :, 1] == True, 255, 0)
        #assert np.array_equal(img, img1)
        
        if show:
            cv2.imshow('Mask', img)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                break
print(f'Processed {i+1} imgs')

