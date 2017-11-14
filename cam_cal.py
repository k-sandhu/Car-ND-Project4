import os
import pickle
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import exposure

def cal_wrapper():
    cal_folder = './camera_cal/'
    test_folder = './test_images/'
    output_folder = './output_images/'
    cal_file_list = [f for f in os.listdir(cal_folder) if f.find('.jpg') != -1]
    test_file_list = [f for f in os.listdir(test_folder) if f.find('.jpg') != -1]

    images = []
    objpoints = []
    imgpoints = []
    nx = 9
    ny = 6

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for fname in cal_file_list:
        ret, corners, img = detect_corners(cal_folder + fname, nx, ny)
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)
            images.append(img)

    ret, mtx, dist, rvecs, tvecs = cal_camera(objpoints, imgpoints, images[0])
    for fname, img, corners in zip(cal_file_list, images, imgpoints):
        undistorted, warped, M = corners_unwarp(img, nx, ny, corners, mtx, dist)

        fname = re.split('.jpg', fname)[0]
        cv2.imwrite(output_folder + fname + '_undistorted.jpg', undistorted)
        cv2.imwrite(output_folder + fname + '_wraped.jpg', warped)

    for fname in test_file_list:
        img = cv2.imread(test_folder+fname)

        undistorted = undistort(img, mtx, dist)

        fname = re.split('.jpg', fname)[0]
        cv2.imwrite(output_folder + fname + '_undistorted.jpg', undistorted)

    pickle.dump(objpoints, open(cal_folder + 'objpoints.pkl', 'wb'))
    pickle.dump(imgpoints, open(cal_folder + 'imgpoints.pkl', 'wb'))
    pickle.dump(mtx, open(cal_folder + 'mtx.pkl', 'wb'))
    pickle.dump(dist, open(cal_folder + 'dist.pkl', 'wb'))


def detect_corners(fname, nx, ny):
    # Make a list of calibration images
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    return ret, corners, img


def cal_camera(objpoints, imgpoints, img):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1][1:3], None, None)
    return ret, mtx, dist, rvecs, tvecs


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def corners_unwarp(img, nx, ny, corners, mtx, dist):
    img_size = (img.shape[1], img.shape[0])
    offsety, offsetx = int(img_size[0] * .1), int(img_size[1] * .1)

    undistorted = undistort(img, mtx, dist)

    # Define 4 source points src = np.float32([[,],[,],[,],[,]])
    src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])

    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[offsety, offsetx], [img_size[0] - offsety, offsetx],
                      [img_size[0] - offsety, img_size[1] - offsetx], [offsety, img_size[1] - offsetx]])

    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undistorted, M, img_size, flags=cv2.INTER_LINEAR)
    return undistorted, warped, M

if __name__ == '__main__':
    cal_wrapper()