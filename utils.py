import os
import shutil
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip

def get_binary(img, mtx, dist, s_thresh=(55, 255), sx_thresh=(5, 90), r_type ='cb'):
    """
    Performs binary transformation on undistorted image using color gradient threshold in HLS space and
    Sobel gradient threshold along the x-axis.

    :param img: Image to be transformed
    :param mtx: Camera transformation matrix
    :param dist: Camera distortion coefficient
    :param s_thresh: Threshold color channel
    :param sx_thresh: Sobel threshold x gradient
    :param r_type: 'cb' to return binary and undistorted original image else returns s_binary, sxbinary too.
    :return: if r_type is 'cb' return binary and undistorted original image else returns s_binary, sxbinary too
    """
    img = np.copy(img)

    undist = undistort(img, mtx, dist) # undistort the image

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS).astype(np.float)

    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.zeros_like(s_channel)
    color_binary[(sxbinary == 1) & (s_binary == 1)] = 1

    if r_type is 'cb':
        return color_binary, undist
    else:
        return (color_binary, undist, s_binary, sxbinary)

def undistort(img, mtx, dist):
    """
    Undistorts the image.
    :param img: Image to be transformed
    :param mtx: Camera transformation matrix
    :param dist: Camera distortion coefficient
    :return: Undistorted image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

def perspective_transform(img, y1=445, d=100, reverse ='n'):
    """
    Performs a perspective on the image. Reverse transform is performed if reverse is 'y'.
    :param img: Undistorted image
    :param y1: y cutoff
    :param d: distance between x1 and x2 at y cutoff
    :param reverse: Does reverse transform if reverse is 'y'
    :return: Tuple of warped imaged and transformation matrix
    """
    img_size = (img.shape[1], img.shape[0])
    x1 = img.shape[1] // 2

    src = np.float32([[x1 - d // 2, y1], [x1 + d // 2, y1], [0, 720], [1280, 720]])
    dst = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])

    if reverse is not 'y':
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    else:
        d=d+10
        y1=y1+5
        src = np.float32([[x1 - d // 2, y1], [x1 + d // 2, y1], [0, 720], [1280, 720]])
        dst = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
        M = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M

def get_pixels(binary_warped):
    """
    Uses the binay_warped image to detect locations of pixels.
    :param binary_warped: Binarized and warped image of lane lines
    :return: Tuple of line pixel locations leftx, lefty, rightx, righty
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 12
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

def fit_lines(pixels):
    """
    Fits a second degree polynomial to pixel locations of the lanes line.
    :param pixels: Tuple of pixel locations leftx, lefty, rightx, righty
    :return: Tuple of coefficients of left and right lane lines
    """
    leftx, lefty, rightx, righty = pixels[0], pixels[1], pixels[2], pixels[3]

    # Polynomial is only fit if lane pixels were detected
    if len(leftx) == 0:
        left_fit = None
    else: left_fit = np.polyfit(lefty, leftx, 2)

    if len(rightx) == 0:
        right_fit = None
    else: right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def draw_lines(undist, warped, left_fit, right_fit):
    """
    Draws lane lines using the polynomial coefficients.
    :param undist: Undistorted image
    :param warped: Warped lane lines image
    :param left_fit: Coefficients of left lane
    :param right_fit: Coefficients of right lane
    :return: Image with lane lines overlayed
    """
    # Create an image to draw the lines on
    x, Minv = perspective_transform(warped, reverse='y')
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

def add_info(img, left_fit, right_fit):
    """
    Add curvature and centre text to the image
    :param img: Original image
    :param left_fit: Coefficients of left lane
    :param right_fit: Coefficients of right lane
    :return: Tupule of image with curvature/distance info and curvature
    """
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    # get pixel data from line coefficients
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    y_eval = np.max(ploty)
    py = ploty * ym_per_pix
    px = leftx * xm_per_pix

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Get left and right lane positions at y max -1
    left_position = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval  + left_fit[2]
    right_position = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval  + right_fit[2]

    # Average the curvature of left and right lanes
    curvature = (left_curverad+right_curverad)/2

    # Calculate the centre of the lane
    lane_centre = np.int16(np.absolute(left_position + right_position)// 2)
    frame_centre = img.shape[1]//2

    # Calculate distance of lane centre from the centre of the image
    from_centre = np.absolute((np.absolute(lane_centre)-frame_centre))*xm_per_pix

    curvature_text = 'Curvature is: {:1.2f}.'.format(curvature)
    from_centre_text = 'Dist from centre: {:1.2f} m.'.format(from_centre)

    x1, y1 = np.int16(np.absolute(left_position)), 719
    x2, y2 = np.int16(np.absolute(right_position)), 719
    x3, y3 = lane_centre, 719

    # Overlay points to track lane centre and image centre
    cv2.circle(img, (x1, y1), 3, (0, 255, 0), -1)
    cv2.circle(img, (x2, y2), 3, (0, 255, 0), -1)
    cv2.circle(img, (x3, y3), 3, (255, 0, 0), -1)

    font_position1 = (50, 600)
    font_position2 = (50, 650)
    font_scale = .4
    font_thickness = 1

    # Overlay text
    cv2.putText(img, curvature_text, font_position1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness, cv2.LINE_AA)
    cv2.putText(img, from_centre_text, font_position2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness, cv2.LINE_AA)

    return img, curvature

def write_video(project_video_output, output_folder, fps=20):
    """
    Reads images from the directory and outputs a video
    :param project_video_output: Name of output video
    :param output_folder: Location of images
    :param fps: Number of frames per second
    :return: None
    """
    print("Creating video {}, FPS={}".format(project_video_output, fps))
    clip = ImageSequenceClip(output_folder, fps)
    clip.write_videofile(project_video_output)

def clean_output_folder(output_folder):
    """
    Deletes the output images from the previous run
    :param output_folder: Folder where images are saved.
    :return: None
    """
    for root, dirs, files in os.walk(output_folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))