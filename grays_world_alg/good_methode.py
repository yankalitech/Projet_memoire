from __future__ import (
    division, absolute_import, print_function, unicode_literals)

import cv2 as cv
import numpy
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import time


def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def clahe_method(image):
    image_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=3)
    return clahe.apply(image_bw)

# noise removal
def remove_noise(image):
    return cv.medianBlur(image,1)

# thresholding
def thresholding(image):
    return cv.threshold(image, 125, 200, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

# dilation
def dilate(image):
    kernel = np.ones((2,2),np.uint8)
    return cv.dilate(image, kernel, iterations = 1)

# erosion
def erode(image):
    kernel = np.ones((1,2),np.uint8)
    return cv.erode(image, kernel, iterations = 1)

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((2,2),np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv.Canny(image, 50, 50)


def show(init, original, final):
    print('display')
    # cv.imshow('Temple', final)
    # cv.waitKey(0)
    fig, ax = plt.subplots(nrows=3, figsize=(10, 20))
    ax[0].imshow(init)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(original)
    ax[1].set_title('Sobel filtered')
    ax[1].axis('off')
    ax[2].imshow(final)
    ax[2].set_title('Sobel filtered')
    ax[2].axis('off')
    cv.imwrite('filename.jpg', final)
    fig.tight_layout()
    plt.show()
    cv.imshow('Temple', final)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Insert any filename with path
img = cv.imread("../path_img/00000.ppm")

def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result

def avg_gray(a,b,c):
    return a/3 + b/3 + c/3


def other_gray_world(result):
    im_new = np.zeros(result.shape, dtype=int)
    # result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_r = np.average(result[:, :, 0])
    avg_g = np.average(result[:, :, 1])
    avg_v = np.average(result[:, :, 2])
    k = avg_r/3 + avg_g/3 + avg_v/3
    print("mean rgb", k/avg_r, k/avg_g, k/avg_v)
    im_new[:, :, 0] = result[:, :, 0] * (k / avg_r)
    im_new[:, :, 1] = result[:, :, 1] * (k / avg_g)
    im_new[:, :, 2] = result[:, :, 2] * (k / avg_v)
    im_new[im_new > 255] = 255
    im_new = np.uint8(im_new)
    print('zeros array', im_new.shape)
    # im_new[:, :, 0] = result[:, :, 0] * (avg_gray(result[:, :, 0], result[:, :, 1], result[:, :, 2]) / result[:, :, 0])
    # im_new[:, :, 1] = result[:, :, 1] * (avg_gray(result[:, :, 0], result[:, :, 1], result[:, :, 2]) / result[:, :, 1])
    # im_new[:, :, 2] = result[:, :, 2] * (avg_gray(result[:, :, 0], result[:, :, 1], result[:, :, 2]) / result[:, :, 2])
    return im_new

def clahe_method(img):
    # Reading the image from the present directory
    # image = cv.imread(img)
    # Resizing the image for compatibility
    #image = cv2.resize(image, (500, 600))

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    print('dtype imqge', img.dtype)
    img1 = img.copy()
    image_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv.createCLAHE(clipLimit=192, tileGridSize=(10, 10))
    final_img = clahe.apply(image_bw)
    print('shqpe', final_img.shape[0])
    resized = cv.resize(final_img, (final_img.shape[1], final_img.shape[0]), interpolation=cv.INTER_AREA)
    #b = cv.resize(final_img.astype('float'), final_img.shape, interpolation=cv.INTER_CUBIC)
    #fg = cv.bitwise_or(img1, img1, mask=b)

    # Ordinary thresholding the same image
    #_, ordinary_img = cv.threshold(img, 155, 255, cv.THRESH_BINARY)

    # Showing all the three images
    result = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    bitwiseAnd = img1

    bitwiseAnd[:, :, 0] = result[:, :, 0]/2 + img1[:, :, 0]/2
    bitwiseAnd[:, :, 1] = result[:, :, 1]/2 + img1[:, :, 1]/2
    bitwiseAnd[:, :, 2] = result[:, :, 2]/2 + img1[:, :, 2]/2

    return bitwiseAnd
def mser_methode (img):
    # img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = img.copy()
    mser = cv.MSER_create()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dilate1 = dilate(gray)

    vis = gray.copy()

    #detect regions in gray scale image
    regions, elm = mser.detectRegions(dilate1)
    print('mask shape',elm)
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv.polylines(vis, hulls, 1, (0, 255, 0))

    cv.imshow('img', vis)

    cv.waitKey(0)

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    print('mask shape', mask.shape)
    #cv.imshow('img mask', mask)
    #
    cv.waitKey(0)
    for contour in hulls:
        cv.drawContours(img, contour, 3, (0, 255, 0), 3)

    #this is used to find only text regions, remaining are ignored
    text_only = cv.bitwise_and(img, img, mask=mask)

    cv.imshow("text only", img)

    cv.waitKey(0)

    return text_only

# def white_balance_loops(img):
#     result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
#     # show(result)
#     print('result', result)
#     avg_a = np.average(result[:, :, 1])
#     avg_b = np.average(result[:, :, 2])
#     for x in range(result.shape[0]):
#         for y in range(result.shape[1]):
#             l, a, b = result[x, y, :]
#             # fix for CV correction
#             print('result passe')
#             l *= 100 / 255.0
#             result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
#             result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
#     result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
#     return result
# other_gray_world(img)
start = time.time()
#final = np.hstack((img, white_balance(img)))

final = other_gray_world(img)

final1 = clahe_method(final)
final3 = mser_methode(final1)
end = time.time()
print('time = ', end - start)
final2 = np.hstack((final, final3))
show(img, final, final2)
# cv.imwrite('result.jpg', final)

#final2 = mser_methode(final1)
#show(final2)