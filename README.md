#### Self-Driving Car Nanodegree Project 2: Advanced Lane Finding Write-up
-------------------------------------------------------------------------
The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image to birds eye view.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/571/view) points individually and describe how I addressed each point in my implementation.

__________________________________________________________________________________________


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Code of this transformation is included below.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

```python
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
```


I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.


```python
def cal_camera(objpoints, imgpoints, img):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1][1:3], None, None)
    return ret, mtx, dist, rvecs, tvecs
```


I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:
[//]: # (Image References)

[image1]: ./camera_cal/calibration10.jpg "Original image"
[image2]: ./output_images/calibration1_undistorted.jpg "Undistorted"
[image3]: ./output_images/test1undistored.jpg "Undistored Test Image"
[image4]: ./output_images/straight_lines2_binary.jpg "Color and Sobel Transform Test Image"
[image5]: ./output_images/straight_lines2_bin_warp.jpg "Binary warped image 1"
[image6]: ./output_images/straight_lines1_bin_warp.jpg "Binary warped image 2"

![alt text][image1]
![alt text][image2]


-------------------------------------------------------------------------------------------

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Image below provides an example of distortion corrected image. Grid lines help us see the difference between distorted and undistorted images.
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 7 through 50 in `utils.py`). I tried various thresholding
 limits to get to the result. Sobel gradients are calculated using `cv2.Sobel()` after transforming the image to HLS color space. I found that limits (5,90) gave
  an optimal result. Threshold value of (55,255) was used for S channel thresholding. Here's an example of my output for this step.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 62 through 87 in the file `utils.py`.  I hardcoded the source and destination points in the following manner:

```python
x1 = img.shape[1] // 2
y1 = 445
d = 100
src = np.float32([[x1 - d // 2, y1], [x1 + d // 2, y1], [0, 720], [1280, 720]])
dst = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 590, 445      | 0, 0        |
| 690, 445      | 1280, 0      |
| 0, 720     | 0, 720      |
| 1280, 720      | 1280, 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][image5]
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I did the following steps to fit the lane lines and smooth out their location over multiple frames.
1. Extracted line pixels from binary warped images generated in the last step. Lines 89-164 in `utils.py`.
2. If the number of pixels identified for each of the left and right lanes were less than 2000, I borrowed them from the previous frames till the number of frames available to draw the lanes lines was more than 2000. Maximum number of frames used for this purpose was 4. This amounts to a combined time of 0.13 seconds for a 30 fps video. Lines 15-67 in `line.py`.
3. Then I fitted a polynomial curve to these pixels. Lines 166-83 in `utils.py`.
4. If the change in polynomial coefficients was more that 12% from the previous frame, I discarded this coefficient and used the coefficient from the previous frame. Lines 69-99 in `line.py`.
5. There coefficients were then smoothed out over 10 frames.

These steps helped smooth the location of lines while adjusting to change in line locations.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 217 through 284 in my code in `utils.py`. Curvature is calculated with the formulae
R(curve) =(1+(2Ay+B)^2)^3/2 / ∣2A∣ where A, B and C are coefficients of the polynomial. Following code calculates the value.

```python
left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
```

Position of the vehicle with respect to centre is calculated is calculated by measuring the distance of centre of the image frame from the centre of the lane. Centre of the lane is calculated as the mid-point of left and right lane lines. Following code found in lines 247 to 259 in `utils.py`.

```python
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
```

I added a blue dot to the video to tracks center of the lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 7 through 38 in my code in `main.py` in the function `pipeline()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./videos/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that finding an optimal way to smooth lane locations was the biggest challenge. Smoothing of location pixels and lane coefficients provides benefits by providing a memory for the system,
but this can also slow down the rate at which new information is incorporated. I tried many different methods to get to the final solution. At the end, I think I am over smoothing the results. This can slow down the system by increasing memory requirement.

Significant improvements can be made to the project. I will outline some of the obvious ones below.
1. Parameter tuning - Parameters for color threshold, Soble gradient threshold, n_windows for finding line centers, color spaces etc can be tuned to achieve significantly better performance. Additionally, parameters for lines coefficient smoothing, pixel borrowing etc can also be tuned.
2. Better information sharing between frames - I implemented information sharing between frames using pixel borrowing, rejecting coefficients that were different from previous frame and smoothing out coefficient parameters. Other methods that can be used are -
- curvature information from previous frames to smooth out the coefficients. This could be useful as lane curvature is smooth and does not change significantly from frame to frame. If the lanes are curve, car drives slowly which slows the rate of change.
- using slope of left lane to smooth out slope of right lane and vice versa. Left and right lanes always have the same distance between them. This can be used to predict the location of the lanes even if location of 1 lane can be reasonable established.
3. Using different color spaces - Using another color space like HSV along with HLS can be used to better identify lanes or increase estimation confidence.

