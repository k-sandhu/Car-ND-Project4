from utils import *

class Line():
    """
    Line object to track metrics
    """
    def __init__(self):
        self.recent_xfitted = [] #x values of the last n fits of the line
        self.current_fit = [np.array([False])] #polynomial coefficients for the most recent fit
        self.radius_of_curvature = None #radius of curvature of the line in some units
        self.recent_curvature = [] #radius of curvature of the line in some units
        self.diffs = [''] #difference in fit coefficients between last and new fits
        self.pixels = []

    def track_pixels(self, leftx, lefty, rightx, righty):
        """
        Track pixels in the current image. If number of pixels is less that min_num, pixels
        are borrowed from the previous 4 frames.
        :param leftx: Left lane pixel x locations
        :param lefty: Left lane pixel y locations
        :param rightx: Right lane pixel x locations
        :param righty: Right lane pixel y locations
        :return: None
        """
        self.pixels.append([leftx, lefty, rightx, righty])

        min_num = 2000
        if len(leftx) < min_num:
            self.borrow_pixels('left', min_num)
        if len(rightx) < min_num:
            self.borrow_pixels('right', min_num)


    def borrow_pixels(self, side, min_num):
        """
        Borrows pixels from the previous frames.
        :param side: Which side to borrow pixels from
        :param min_num: Number of pixels threshold
        :return: None
        """
        if side is 'left':
            current_pixels_leftx = self.pixels[-1][0]
            current_pixels_lefty = self.pixels[-1][1]

            for index, values in enumerate(self.pixels[-5:-1]):
                current_pixels_leftx = np.concatenate([current_pixels_leftx, values[0]],axis=0)
                current_pixels_lefty = np.concatenate([current_pixels_lefty, values[1]], axis=0)

                if current_pixels_leftx.shape[0] > min_num:
                    break

            self.pixels[-1][0] = current_pixels_leftx
            self.pixels[-1][1] = current_pixels_lefty

        if side is 'right':
            current_pixels_rightx = self.pixels[-1][2]
            current_pixels_righty = self.pixels[-1][3]

            for index, values in enumerate(self.pixels[-5:-1]):
                current_pixels_rightx = np.concatenate([current_pixels_rightx, values[2]], axis=0)
                current_pixels_righty = np.concatenate([current_pixels_righty, values[3]], axis=0)

                if current_pixels_rightx.shape[0] > min_num:
                    break

            self.pixels[-1][2] = current_pixels_rightx
            self.pixels[-1][3] = current_pixels_righty

    def set_recent_xfitted(self, fits):
        """
        Track the current fit coefficients. Compare them to previous fit coefficients. If a coefficient
        is more that 12% off, reject it and used coefficient from the previous frame.
        :param fits: List of fit coefficients
        :return: None
        """
        self.current_fit = fits
        if len(self.recent_xfitted) > 0:
            self.diffs.append([(recent - current)/recent for recent, current in zip(self.recent_xfitted[-1], self.current_fit)])

            bool_diffs_left = np.absolute(self.diffs[-1][0] > .12)
            bool_diffs_right = np.absolute(self.diffs[-1][1] > .12)

            for index, value in enumerate(bool_diffs_left):
                if value is True:
                    self.current_fit[0][index] = self.mean_xfitted(n=10)[0][index]

            for index, value in enumerate(bool_diffs_right):
                if value is True:
                    self.current_fit[1][index] = self.mean_xfitted(n=10)[1][index]

        # Use mean of previous 10 frames if no coefficients were detected.
        if self.current_fit[0] is None and self.current_fit[1] is None:
            self.current_fit = self.mean_xfitted(n=10)
        elif self.current_fit[0] is None:
            self.current_fit[0] = self.mean_xfitted(n=10)[0]
        elif self.current_fit[1] is None:
            self.current_fit[1] = self.mean_xfitted(n=10)[1]

        self.recent_xfitted.append(self.current_fit)

    def get_last_xfitted(self, n=1):
        """
        Return coefficients from the last n frames
        :param n: Number of frames
        :return: Coefficients from the last n frames
        """
        if len(self.recent_xfitted) < n:
            return self.recent_xfitted
        else: return self.recent_xfitted[-n:]

    def mean_xfitted(self, n=1):
        """
        Return coefficient mean of the last n fits
        :param n: Number of frames
        :return: Mean of coefficients from the last n frames
        """
        return np.mean(self.get_last_xfitted(n),axis=0)
