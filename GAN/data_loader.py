import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# The DataLoader class
class DataLoader():
    # Initialization method
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    # Method to load the training data
    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('drive/app/pics/*')

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            print(img_path)
            img = self.imread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = scipy.misc.imresize(img, self.img_res)
            img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    # Method to read an image given a path
    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
