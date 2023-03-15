import numpy as np
import cv2
import os

save = False

dir_path = '/data/jk/sentinel/subscenes'

for i, path in enumerate(os.listdir(dir_path)):
    file_path = os.path.join(dir_path, path)
    if os.path.isfile(file_path):
        print(file_path)
        subscene = np.load(file_path)
        bgr = subscene[...,[1,2,3]]
        if save:
            bgr = cv2.normalize(bgr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(f'subscene_{i}.jpg', bgr)
        cv2.imshow('Subscene', bgr)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
print(f'Processed {i+1} imgs')
