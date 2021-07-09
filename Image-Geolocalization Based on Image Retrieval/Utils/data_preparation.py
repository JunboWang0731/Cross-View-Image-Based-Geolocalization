import numpy as np
import cv2
import os
import math

input_dir = '/home/junbo/ANU_data_small/streetview/'
output_dir = '/home/junbo/ANU_data_small/streetview_purepolar/'

def Hybrid_Perspective_Mapping(img, s, fov):
    '''
    Transform original panoramic image to "birds eye view"-like image
    :param img: the original panoramic image
    :param s: the width and height of transformed image
    :param fov: the field of view of top-down image
    :return: "birds eye view"-like image with width s
    '''

    # Extend height for images in CVUSA dataset to make it satisfy the ideal projection relation
    if img.shape[1] == 1232:
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

    # alpha is the parameter to make sure that there are less sky area in image
    alpha = 0.8 * 1200
    R = np.maximum(R, alpha / s * r)

    Py = H - R
    Px = W / (2 * math.pi) * theta

    Px = Px.astype('float32')
    Py = Py.astype('float32')
    dst = cv2.remap(img, Px, Py, cv2.INTER_LINEAR)

    return dst

def HPM():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(input_dir)
    print('ready to do HPM and resize on ground-level panoramic images...')
    print('     output_dir: ', output_dir)
    for img in images:
        origin = cv2.imread(input_dir + img)
        transformed = Hybrid_Perspective_Mapping(origin, 1200, 0.95)
        transformed = cv2.resize(transformed, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_dir+img, transformed)
    print('finished!')

if __name__ == '__main__':
    HPM()