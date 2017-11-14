import pickle
from datetime import datetime
from skimage.io import imsave
from line import Line
from utils import *

def pipeline(image, line, mtx, dist):
    """
    Primary pipeline that calls all the required methods and returns a processed image to
    the main method.
    1. Undistort's the image and converts the lane area into a binary image.
    2. Warps the binary lane image
    3. Extracts pixels where lanes are detected
    4. Fits second order polynomial to the pixel data
    5. Draws lane lines on the undistorted image
    6. Add curvature and centre data to the image
    :param image: Images extracted from the video
    :param line: Line object used to track lane line stats
    :param mtx: Camera matrix
    :param dist: Distortion coefficient
    :return: Image with lane line area layered on input image
    """

    binary, undist = get_binary(image[:, :, :], mtx, dist, r_type ='cb')
    binary_warped, M = perspective_transform(binary)
    leftx, lefty, rightx, righty = get_pixels(binary_warped)
    line.track_pixels(leftx, lefty, rightx, righty) # track pixel found in the current image

    left_fit, right_fit = fit_lines(line.pixels[-1])
    line.set_recent_xfitted([left_fit, right_fit]) # track coefficients of left and right fit
    fits = line.mean_xfitted(10) #average fits over the last 10 frames. 1/3rd of a second
    left_fit, right_fit = fits[0], fits[1]

    new_warp = draw_lines(undist, binary_warped, left_fit, right_fit)
    new_warp, curvature = add_info(new_warp, left_fit, right_fit)

    line.radius_of_curvature = curvature # Track curvature
    return new_warp

if __name__ == "__main__":
    output_folder = './videos/output/'
    project_video_output = './videos/project_video_output.mp4'
    project_video = "./videos/project_video.mp4"
    cal_folder = './camera_cal/'
    mtx = pickle.load(open(cal_folder + 'mtx.pkl', 'rb')) # Load camera matrix
    dist = pickle.load(open(cal_folder + 'dist.pkl', 'rb')) # Load distortion coefficient

    vidcap = cv2.VideoCapture(project_video) # Create VideoCapture object to extract frames
    success, image = vidcap.read() # reads one frame at a time. Success if true if a frame is found
    count = 0
    line = Line() # create line object

    clean_output_folder(output_folder) # deleted previous transformed images stored in the output folder

    while True:
        success, image = vidcap.read()
        print('Frame Number: ', count+1) # Print out frame number

        # Break when a frame is not read. Breaks at the end of the video
        if success is False:
            break

        # Call to main pipeline
        image = pipeline(image, line, mtx, dist)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        timestamp = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]) # save with timestamp and frame number as file name

        imsave((output_folder + "frame-{0}-{1}.jpg").format(timestamp,count), image) # save frame as a JPEG file
        count += 1 # Track frame number

    # Read all written images and save as a video
    write_video(project_video_output, output_folder)