from __future__ import division
import json
import csv
from .transforms import transform_preds, get_affine_transform, affine_transform
from .landmark_statistics import LandmarkStatistics
import os
import numpy as np
import cv2


def get_center_scale(w, h, aspect_ratio=1.0, scale_mult=1.25):
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = w * 0.5
    center[1] = h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale

def load_csv(file_name, num_landmarks, dim):
    landmarks_dict = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            landmarks = []
            num_entries = dim * num_landmarks + 1
            assert num_entries == len(
                row), 'number of row entries ({}) and landmark coordinates ({}) do not match'.format(num_entries,
                                                                                                     len(row))
            # print(len(points_dict), name)
            for i in range(1, dim * num_landmarks + 1, dim):
                # print(i)
                if dim == 2:
                    coords = np.array([float(row[i]), float(row[i + 1])], np.float32)
                elif dim == 3:
                    coords = np.array([float(row[i]), float(row[i + 1]), float(row[i + 2])], np.float32)
                    # landmark = Landmark(coords)
                landmarks.append(coords)
            landmarks = np.array(landmarks)
            landmarks_dict[id] = landmarks
    return landmarks_dict



def cal_deo_correct(kpt_json, img_size):
    DATASET_PATH = '/nfs/usrhome/mkfmelbatel/eez194/NFDP/data/Cephalometric/'
    IMG_PREFIX = 'RawImage/ALL'
    ANN = '400_senior'
    ANN2 = '400_junior'
    gt_path = os.path.join(DATASET_PATH, ANN)
    gt_path2 = os.path.join(DATASET_PATH, ANN2)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_data = kpt_json
    kpt_num = 19  # Number of keypoints
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
    spacing = [0.1, 0.1]  # Spacing in mm per pixel
    print("Number of data points:", len(kpt_data))
    landmark_dist = []
    landmark_statistic = LandmarkStatistics()

    for i in range(len(kpt_data)):
        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']

        # Load ground truth annotations
        annoFolder = os.path.join(gt_path, img_name[:-4] + '.txt')
        annoFolder2 = os.path.join(gt_path2, img_name[:-4] + '.txt')
        pts1, pts2 = [], []
        with open(annoFolder, 'r') as f:
            lines = f.readlines()
            for line in lines[:kpt_num]:
                coordinates = [int(v) for v in line.strip().split(',')]
                pts1.append(coordinates)
        with open(annoFolder2, 'r') as f:
            lines = f.readlines()
            for line in lines[:kpt_num]:
                coordinates = [int(v) for v in line.strip().split(',')]
                pts2.append(coordinates)

        # Average the senior and junior annotations
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        gt_kpt = (pts1 + pts2) / 2.0

        # Read the image to get its dimensions
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]

        scale = np.array([img_h, img_w]) * 1.25
        center = np.array([img_w * 0.5, img_h * 0.5])

        # Calculate Euclidean distances for each keypoint
        for j in range(kpt_num):
            pred_x = kpt_coord[j * 3]
            pred_y = kpt_coord[j * 3 + 1]
            pred_pt = transform_preds(np.array([pred_x, pred_y]), center, scale, [kpt_w, kpt_h])

            gt_x, gt_y = gt_kpt[j][:2]

            # Store predicted and ground truth points
            pred_pts.append((pred_pt[0], pred_pt[1]))
            gt_pts.append((gt_x, gt_y))

            # Compute Euclidean distance in pixels
            landmark_dist.append(np.sqrt((pred_pt[0] - gt_x) ** 2 + (pred_pt[1] - gt_y) ** 2))

        landmark_statistic.add_landmarks(image_id=img_name, predicted=pred_pts, groundtruth=gt_pts, spacing=spacing)
    overview_string = landmark_statistic.get_overview_string([2.0, 2.5, 3.0, 4.0])
    pe_mean, pe_std, pe_median = landmark_statistic.get_pe_statistics()

    return overview_string, pe_mean

def cal_deo_hand_correct(kpt_json, img_size):
    DATASET_PATH = '/nfs/usrhome/mkfmelbatel/eez194/NFDP/data/hand/'
    IMG_PREFIX = 'Images'
    ANN = 'all.csv'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_data = kpt_json
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
    image_size_in = [512, 512]
    kpt_num = 37
    gt_landmark = load_csv(gt_path, kpt_num, dim=2)
    landmark_dist = []
    landmark_statistic = LandmarkStatistics()
    print(len(kpt_data))
    for i in range(len(kpt_data)):

        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        # load gt ann
        gt_kpt = gt_landmark[img_name]

        img = cv2.imread(os.path.join(data_path, img_name + '.jpg'), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)

        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        # need to check the scale_multi

        scale = np.array([img_h, img_w]) * 1.25
        center = np.array([img_w * 0.5, img_h * 0.5])

        # center_beta = np.array([kpt_w_beta * 0.5, kpt_h_beta * 0.5])
        for j in range(kpt_num):
            pred_x = kpt_coord[j * 3]
            pred_y = kpt_coord[j * 3 + 1]
            coord_draw = transform_preds(np.array([pred_x, pred_y]), center, scale, [kpt_w, kpt_h])
            gt_kpts = gt_kpt[j][0:2]
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpts[0]), int(gt_kpts[1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpts[0]) ** 2 + (coord_draw[1] - gt_kpts[1]) ** 2))

        landmark_statistic.add_landmarks(image_id=img_name, predicted=pred_pts, groundtruth=gt_pts,
                                         normalization_factor=50, normalization_indizes=[0, 4])
    overview_string = landmark_statistic.get_overview_string([2.0, 4.0, 10.0])
    pe_mean, pe_std, pe_median = landmark_statistic.get_pe_statistics()
    return overview_string, pe_mean


def cal_deo_chest_correct(kpt_json, img_size):
    DATASET_PATH = '/nfs/usrhome/mkfmelbatel/eez194/NFDP/data/chest/'
    IMG_PREFIX = 'Images'
    ANN = 'all.csv'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_data = kpt_json
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
    image_size_in = [512, 512]
    kpt_num = 6
    gt_landmark = load_csv(gt_path, kpt_num, dim=2)
    landmark_dist = []
    landmark_statistic = LandmarkStatistics()
    print(len(kpt_data))
    for i in range(len(kpt_data)):

        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        # load gt ann
        gt_kpt = gt_landmark[img_name]

        img = cv2.imread(os.path.join(data_path, img_name + '.png'), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)

        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        # need to check the scale_multi
        center, scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.25)
        trans = get_affine_transform(center, scale, 0, image_size_in)
        for j in range(kpt_num):
            coord_draw = np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])])
            gt_kpts = affine_transform(gt_kpt[j][0:2], trans)
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpts[0]), int(gt_kpts[1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpts[0]) ** 2 + (coord_draw[1] - gt_kpts[1]) ** 2))

        landmark_statistic.add_landmarks(image_id=img_name, predicted=pred_pts, groundtruth=gt_pts, spacing=[1, 1])
    overview_string = landmark_statistic.get_overview_string([2.0, 2.5, 3.0, 4.0])
    pe_mean, pe_std, pe_median = landmark_statistic.get_pe_statistics()

    return overview_string, pe_mean


def cal_deo_leg_correct(kpt_json, img_size):
    DATASET_PATH = '/nfs/usrhome/mkfmelbatel/eez194/NFDP/data/leg/'
    IMG_PREFIX = 'Images'
    ANN = 'all.csv'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    meta_json_path = os.path.join(DATASET_PATH, 'meta_json')
    kpt_data = kpt_json
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
    image_size_in = [512, 512]
    kpt_num = 26
    gt_landmark = load_csv(gt_path, kpt_num, dim=2)
    landmark_dist = []
    landmark_statistic = LandmarkStatistics()
    print(len(kpt_data))
    for i in range(len(kpt_data)):

        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        # load gt ann
        gt_kpt = gt_landmark[img_name]

        img = cv2.imread(os.path.join(data_path, img_name + '.png'), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)

        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        # need to check the scale_multi

        scale = np.array([img_h, img_w]) * 1.25
        center = np.array([img_w * 0.5, img_h * 0.5])

        meta_json = os.path.join(meta_json_path, img_name + '_meta.json')
        with open(meta_json, 'r') as file:
            data = json.load(file)
            # Extract the spacing if present
            spacing_tuple = list(data['spacing'])  # Convert list to tuple for immutability


        # center_beta = np.array([kpt_w_beta * 0.5, kpt_h_beta * 0.5])
        for j in range(kpt_num):
            pred_x = kpt_coord[j * 3]
            pred_y = kpt_coord[j * 3 + 1]
            coord_draw = transform_preds(np.array([pred_x, pred_y]), center, scale, [kpt_w, kpt_h])
            gt_kpts = gt_kpt[j][0:2]
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpts[0]), int(gt_kpts[1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpts[0]) ** 2 + (coord_draw[1] - gt_kpts[1]) ** 2))

        landmark_statistic.add_landmarks(image_id=img_name, predicted=pred_pts, groundtruth=gt_pts, spacing=spacing_tuple)

    overview_string = landmark_statistic.get_overview_string([2.0, 2.5, 3.0, 4.0])
    pe_mean, pe_std, pe_median = landmark_statistic.get_pe_statistics()
    return overview_string, pe_mean



