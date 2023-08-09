import os

import numpy as np
import cv2 as cv

from dataset import HVMDataset


def get_sift_features(num_kps=5):
    """
    Get SIFT features from the HVM dataset.
    return all_features: a numpy array of shape (num_images, num_kps * 128)
    """

    dataset = HVMDataset(split='all')
    sift = cv.SIFT_create(nfeatures=2*num_kps)

    all_features = []
    for i in range(len(dataset)):
        img_fn = dataset.normed_data_frame.loc[i, 'image_file_name']
        img_fn = os.path.join(dataset.root_dir, img_fn)

        img = cv.imread(img_fn)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(gray, None)
        feature_vct = des.reshape(1, -1)[:, :num_kps*128]
        assert feature_vct.shape[1] == num_kps * 128, f'feature length: {feature_vct.shape[1]} != {num_kps*128}'
        all_features.append(feature_vct)

        if i % 100 == 0:
            img = cv.drawKeypoints(gray, kp, img,
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite(os.path.join('./data/hvm_dataset', f'img{i}_sift_keypoints.jpg'), 
                       img)

    all_features = np.concatenate(all_features, axis=0)
    all_features = (all_features - all_features.mean(axis=0)) / all_features.std(axis=0)
    return {'SIFT': all_features}, dataset.normed_data_frame
