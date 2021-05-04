import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from collections import defaultdict
from shapely.geometry import MultiPolygon, Polygon


def display_rgb_img(img, display_size=(5,5)):
    plt.figure(figsize=display_size)
    plt.imshow(img)

def display_mask(mask):
    plt.figure(figsize=(5,5))
    # plt.imshow(mask, cmap='gray')
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)

def display_stacked_masks(stacked_masks):
    merged_mask = np.zeros(stacked_masks.shape[:2])
    for i in range(stacked_masks.shape[-1]):
        merged_mask += stacked_masks[:,:,i]
    plt.figure(figsize=(5,5))
    plt.imshow(merged_mask, cmap='gray')
    # plt.imshow(merged_mask, cmap='gray', vmin=0, vmax=1)

def get_patch(data, patch_shape=(256, 256), patch_coord=(0, 0)):
    coord_x_start = patch_coord[0]*patch_shape[0]
    coord_x_end = coord_x_start+patch_shape[0]

    coord_y_start = patch_coord[1]*patch_shape[1]
    coord_y_end = coord_y_start+patch_shape[1]

    data_patch = data[coord_x_start:coord_x_end,
                      coord_y_start:coord_y_end,:]

    return data_patch

def mask_to_polygons(mask, epsilon=10., min_area=10.):
    # Author: Konstantin Lopuhin
    # Source https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def mask_for_polygons(polygons, im_size):
    # Author: Konstantin Lopuhin
    # Source https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def weighted_categorical_crossentropy(weights):
    # Author: Mendi Barel
    # Source: https://stackoverflow.com/questions/59520807/multi-class-weighted-loss-for-semantic-image-segmentation-in-keras-tensorflow
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not tf.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

def calc_iou(y_true, y_pred):
    # Author: Vladimir Iglovikov
    # Source: https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])
    iou = intersection / (sum_ - intersection)
    return iou

def calc_mean_iou(y_true, y_pred):
    # Author: Vladimir Iglovikov
    # Source: https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])
    iou = intersection / (sum_ - intersection)
    return K.mean(iou)

def evaluate_iou(data_gen, model):
    classes_iou = np.zeros(data_gen.num_classes)
    for x, y in data_gen:
        y_pred = model.predict(x)
        classes_iou += calc_iou(y, y_pred)
    return classes_iou/len(data_gen)

def split_list(input_list, split_ratio=(0.7, 0.2, 0.1)):
    split_idxs = []

    idx_pos = 0
    for ratio in split_ratio[:-1]:
        idx_pos += int(len(input_list)*ratio)
        split_idxs.append(idx_pos)

    splited_lists = [input_list[i:j] for i, j in zip([0]+split_idxs, split_idxs+[None])]
    return splited_lists



