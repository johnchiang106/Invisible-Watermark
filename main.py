#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from invisible_watermark import WaterMark
import os

def cut_attack(input_filename=None, input_img=None, output_file_name=None, loc_r=None, scale=None, loc = None):
    if input_filename:
        input_img = cv2.imread(input_filename)

    if loc is None:
        h, w, _ = input_img.shape
        x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])
    else:
        x1, y1, x2, y2 = loc
    output_img = input_img[y1:y2, x1:x2].copy()

    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img

def estimate_crop_parameters(original_file=None, template_file=None, ori_img=None, tem_img=None):
    if template_file:
        tem_img = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)  # template image
    if original_file:
        ori_img = cv2.imread(original_file, cv2.IMREAD_GRAYSCALE)  # image
    scores = cv2.matchTemplate(ori_img, tem_img, cv2.TM_CCOEFF_NORMED)
    ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    ind, score = ind, scores[ind]
    w, h = int(tem_img.shape[1]), int(tem_img.shape[0])
    x1, y1, x2, y2 = ind[1], ind[0], ind[1] + w, ind[0] + h
    return (x1, y1, x2, y2), ori_img.shape, score

def recover_crop(template_file=None, tem_img=None, output_file_name=None, loc=None, image_o_shape=None, shape = None):
    if template_file:
        tem_img = cv2.imread(template_file)  # template image
    (x1, y1, x2, y2) = loc

    img_recovered = np.zeros((image_o_shape[0], image_o_shape[1], 3))

    img_recovered[y1:y2, x1:x2, :] = cv2.resize(tem_img, dsize=(x2 - x1, y2 - y1))

    if output_file_name:
        cv2.imwrite(output_file_name, img_recovered)
    return img_recovered

def salt_pepper_att(input_filename=None, input_img=None, output_file_name=None, ratio=0.01):
    if input_filename:
        input_img = cv2.imread(input_filename)
    input_img_shape = input_img.shape
    output_img = input_img.copy()
    for i in range(input_img_shape[0]):
        for j in range(input_img_shape[1]):
            if np.random.rand() < ratio:
                output_img[i, j, :] = 255
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img

def rot_att(input_filename=None, input_img=None, output_file_name=None, angle=45):
    if input_filename:
        input_img = cv2.imread(input_filename)
    rows, cols, _ = input_img.shape
    M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
    output_img = cv2.warpAffine(input_img, M, (cols, rows))
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img

def gaussian_att(input_filename=None, input_img=None, output_file_name=None, kernel_size=(13, 13), sigma=1):
    # Read the image
    if input_filename:
        input_img = cv2.imread(input_filename)
    # Apply Gaussian blur
    output_img = cv2.GaussianBlur(input_img, kernel_size, sigma)

    # Save the result
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img

originalImageFile = 'pic/2880.jpg'
watermarkFile = 'pic/qrcode.png'
embeddedImageFile = 'output/qr_embedded2.png'
extractedFile = 'output/qr_extracted2.png'
os.chdir(os.path.dirname(__file__))
img = cv2.imread(originalImageFile, flags=cv2.IMREAD_UNCHANGED)
wm = cv2.imread(watermarkFile, flags=cv2.IMREAD_UNCHANGED)
h, w = img.shape[0:2]
iwm = WaterMark(password_wm=1, password_img=1)
iwm.read_img(img = img)
iwm.read_wm(watermarkFile)
embed_img = iwm.embed(embeddedImageFile)
wm_shape = cv2.imread(watermarkFile, flags=cv2.IMREAD_GRAYSCALE).shape
iwm1 = WaterMark(password_wm=1, password_img=1)
iwm1.extract(embeddedImageFile, wm_shape=wm_shape, out_wm_name=extractedFile)
# ----------------------------------

loc_r = ((0.1, 0.1), (0.8, 0.8))
x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])

img_attacked = cut_attack(input_img=embed_img, loc=(x1, y1, x2, y2), output_file_name = 'output/attact_pic.png')


# estimate crop attack parameters:
(x1, y1, x2, y2), image_o_shape, score = estimate_crop_parameters(ori_img=embed_img,tem_img=img_attacked)

# recover from attack:
img_recover = recover_crop(tem_img=img_attacked,
                           loc=(x1, y1, x2, y2), image_o_shape=image_o_shape, output_file_name = 'output/recover_pic.png')

iwm1 = WaterMark(password_wm=1, password_img=1)
iwm1.extract(embed_img=img_recover, wm_shape=wm_shape, out_wm_name='output/att_extract.png')

#------------------------------
img_attacked = salt_pepper_att(input_img=embed_img, ratio=0.05, output_file_name = 'output/attact_pic2.png')
iwm1.extract(embed_img=img_attacked, wm_shape=wm_shape, out_wm_name = 'output/att_extract2.png')
#--------------------------------
angle = 50
img_attacked = rot_att(input_img=embed_img, angle=angle, output_file_name = 'output/attact_pic3.png')
img_recover = rot_att(input_img=img_attacked, angle=-angle, output_file_name = 'output/recover_pic3.png')
iwm1.extract(embed_img=img_recover, wm_shape=wm_shape, out_wm_name = 'output/att_extract3.png')
#------------------------
img_attacked = gaussian_att(input_img=embed_img, output_file_name = 'output/attact_pic4.png',sigma=3)
iwm1.extract(embed_img=img_attacked, wm_shape=wm_shape, out_wm_name = 'output/att_extract4.png')