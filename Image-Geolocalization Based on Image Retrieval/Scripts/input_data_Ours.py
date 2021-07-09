import cv2
import numpy as np
import math

SatLevel = "19"
GrdType = "Aligned"

class InputData:

    img_root = '/media/junbo/Elements/CVM_Dataset/'

    def __init__(self):
        print("Now you are using Ours dataset")
        self.test_list = self.img_root + 'splits/' + SatLevel + "_" + GrdType + '.csv'

        # For Testing
        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                item1 = data[0]
                item2 = data[1]
                self.id_test_list.append([item1, item2, pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)

    # warp ground-level images
    def warp(self, img):
        # s means height and width of dst image, which size is s*s pixels
        # fov means field of view(in radius)
        # Ch means cemera height from image plane(in pixels)
        if img.shape[1] == 1664:  # for CVACT
            s = 1200
        elif img.shape[1] == 2048: # for Ours dataset
            s = 512
        else: # for CVUSA
            s = 512
            img = cv2.copyMakeBorder(img, 168, 224, 0, 0, cv2.BORDER_REPLICATE)

        fov = 0.95 * math.pi
        Ch = s / 2 / np.tan(fov / 2)
        H = img.shape[0]
        W = img.shape[1]

        Y, X = np.meshgrid(np.arange(0, s, 1), np.arange(0, s, 1))

        # theta generate from arctan2 is in range (-pi, pi). theta should add 2*pi for Y > s/2 so it ranged from (0, pi)
        theta = np.arctan2(s / 2 - Y, X - s / 2)
        theta[:, int(s / 2) + 1:s] = theta[:, int(s / 2) + 1:s] + 2 * math.pi
        r = np.sqrt(np.power((s / 2 - Y), 2) + np.power((s / 2 - X), 2))
        R = np.arctan(r / Ch) * H / math.pi

        # alpha is parameter to adjust the border between perspective mapping and polar transform
        alpha = 1200
        R = np.maximum(R, alpha / s * r)

        Py = H - R
        Px = W / (2 * math.pi) * theta

        Px = Px.astype('float32')
        Py = Py.astype('float32')
        dst = cv2.remap(img, Px, Py, cv2.INTER_LINEAR)

        return dst

    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        batch_grd = np.zeros([batch_size, 224, 224, 3], dtype = np.float32)
        batch_sat = np.zeros([batch_size, 224, 224, 3], dtype=np.float32)

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # satellite
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            cv2.imshow('sat', img)
            cv2.waitKey(0)

            img = img.astype(np.float32)
            img = img / 255.0
            img = img * 2.0 - 1.0
            batch_sat[i, :, :, :] = img

            # ground
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][1])

            img = self.warp(img)

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            cv2.imshow('grd', img)
            cv2.waitKey(0)

            img = img.astype(np.float32)
            img = img / 255.0
            img = img * 2.0 - 1.0
            batch_grd[i, :, :, :] = img

        self.__cur_test_id += batch_size

        return batch_sat, batch_grd

    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0

if __name__ == '__main__':
    input_data = InputData()
    batch_sat, batch_grd = input_data.next_batch_scan(16)

