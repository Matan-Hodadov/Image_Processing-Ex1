"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 212372494


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img_matx = mpimg.imread(filename, representation) / 255
    if representation == 1:
        img_matx = np.dot(img_matx, [0.2989, 0.5870, 0.1140])
    return img_matx


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img, cmap='gray')
    plt.show()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    rgb_to_yiq_matrix = np.array([[0.299, 0.587, 0.114],
                                  [0.59590059, -0.27455667, -0.32134392],
                                  [0.21153661, -0.52273617, 0.31119955]])
    return np.dot(imgRGB, rgb_to_yiq_matrix.T)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_to_rgb_matrix = np.array([[1, 0.956, 0.619],
                                  [1, -0.272, -0.647],
                                  [1, -1.106, 1.703]])
    return np.dot(imgYIQ, yiq_to_rgb_matrix.T)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    flag_is_rgb = False
    imgOrig_temp = imgOrig.copy()
    if len(imgOrig.shape) != 2:
        flag_is_rgb = True
        img_org_yiq = transformRGB2YIQ(imgOrig_temp)
        imgOrig_temp = img_org_yiq[:, :, 0].copy()

    imgOrig_temp = imgOrig_temp * 255
    org_hist, bin_edges = np.histogram(imgOrig_temp.ravel(), bins=256, range=[0, 255])
    hist_cumsum = np.cumsum(org_hist)
    # lut = hist_cumsum / max(org_hist) * 255
    lut = hist_cumsum / len(imgOrig_temp.ravel()) * 255
    lutimg = cv2.LUT(imgOrig_temp.astype('uint8'), lut.astype('uint8'))
    img_eq = lutimg

    if flag_is_rgb:
        img_org_yiq[:, :, 0] = lutimg / 255.0
        imgOrig_temp = transformYIQ2RGB(img_org_yiq.copy())
        img_eq = imgOrig_temp

    eq_hist, bins = np.histogram(lutimg.ravel(), bins=256, range=[0, 255])
    return img_eq, org_hist, eq_hist


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    if len(imOrig.shape) != 2:
        im_yiq = transformRGB2YIQ(imOrig.copy())
        im_y = im_yiq[..., 0].copy()  # take only the y chanel
    else:
        im_y = imOrig
    histOrig, bins_edge = np.histogram(im_y.ravel(), bins=256)
    Z, Q = findCenters(histOrig, nQuant, nIter)
    quantize_image_history = [imOrig.copy()]
    error_list = []
    for i in np.arange(len(Z)):
        arrayQuantize = np.array([Q[i][k] for k in np.arange(len(Q[i])) for x in np.arange(Z[i][k], Z[i][k + 1])])
        q_img, error = convertToImg(im_y, histOrig, im_yiq if len(imOrig.shape) == 3 else [], arrayQuantize)
        quantize_image_history.append(q_img)
        error_list.append(error)

    return quantize_image_history, error_list


def find_new_q(z: np.array, image_hist: np.ndarray) -> np.ndarray:
    # q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=image_hist[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
    q = []
    for index in np.arange(len(z)-1):
        q.append(np.average(np.arange(z[index], z[index + 1] + 1), weights=image_hist[z[index]: z[index + 1] + 1]))
    return np.round(q).astype(int)


def find_new_z(q: np.array) -> np.array:
    # z = np.array([round((q[i - 1] + q[i]) / 2) for i in range(1, len(q))]).astype(int)
    z = []
    for index in np.arange(1, len(q)):
        z.append(round((q[index - 1] + q[index]) / 2))
    z = np.array(z)
    z = z.astype(int)
    z = np.concatenate(([0], z, [255]))
    return z


def findCenters(orig_hist: np.ndarray, num_colors: int, n_iter: int) -> (np.ndarray, np.ndarray):
    # z_list = []
    # q_list = []
    z_list, q_list = [], []
    # z = np.arange(0, 256, round(256 / num_colors))
    z = np.arange(0, 256, 256 // num_colors)
    z = np.append(z, [255])
    z_copy = z.copy()
    z_list.append(z_copy)
    q = find_new_q(z, orig_hist)
    q_copy = q.copy()
    q_list.append(q_copy)
    for iter in range(n_iter):
        z = find_new_z(q)
        if (z_list[len(z_list)-1] == z).all():
            break
        z_copy = z.copy()
        z_list.append(z_copy)
        q = find_new_q(z, orig_hist)
        q_copy = q.copy()
        q_list.append(q_copy)
    return z_list, q_list


def convertToImg(imOrig: np.ndarray, histOrig: np.ndarray, yiq_im: np.ndarray, arrayQuantize: np.ndarray)\
        -> (np.ndarray, float):
    quantized_img = np.interp(imOrig, np.linspace(0, 1, 255), arrayQuantize)
    curr_hist, bins = np.histogram(quantized_img, bins=256)
    err = np.sqrt(np.sum((histOrig.astype('float') - curr_hist.astype('float')) ** 2)) / float(
        imOrig.shape[0] * imOrig.shape[1])
    if len(yiq_im):  # if the original image is RGB
        yiq_im[:, :, 0] = quantized_img / 255
        return transformYIQ2RGB(yiq_im), err
    return quantized_img, err