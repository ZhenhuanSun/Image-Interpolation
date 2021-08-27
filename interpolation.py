"""
    Image Interpolation Class
        - Nearest Neighbour Interpolation
        - Bilinear Interpolation
        - Bicubic Interpolation
    Arthor: Zhenhuan(Steven) Sun
"""

import numpy as np
import cv2 # only for loading image and showing image

class interpolation():
    def __init__(self, image_path, scale=1):
        # load image to numpy array
        self.image = cv2.imread(image_path)

        # size of original image
        self.image_width = self.image.shape[0]
        self.image_height = self.image.shape[1]

        # size of output image after rescaling
        self.resized_image_width = self.image_width * scale
        self.resized_image_height = self.image_height * scale

        # the amount of scale
        self.scale = scale

    # resize image using nearest neighbour interpolation
    def nearest_neighbour(self):
        """
            Nearest Neighbour Interpolation
        """
        print("Nearest Neighbour Interpolation:")

        # empty list to store each resized rgb channel
        image_rgb = []

        rgb = "bgr"

        # perform nearest neighbour interpolation for color channel
        for i in range(3):
            print("\tInterpolating {} channel...".format(rgb[i]))
            image_channel = self.image[:, :, i]

            # pre-allcoate space for output image in current color channel
            # make sure the data type is unsigned integer 8 since only uint8 data type are accepted in image file
            image_channel_output = np.zeros((self.resized_image_width, self.resized_image_height), dtype=np.uint8)

            # interpolate image pixel by pixel
            for x in range(self.resized_image_width):
                # compute the corresponding x position in the original image
                x_ori = (x / self.resized_image_width) * self.image_width

                # the amount of interpolation in x axis(eg: x_ori=1.5, then x_interp=0.5, 0.5 from x=1)
                x_interp = x_ori - np.floor(x_ori)

                # find the nearest neighbour of current x
                if x_interp < 0.5:
                    x_int = int(np.floor(x_ori))
                else:
                    x_int = int(np.ceil(x_ori))
                    # if x_int is out of bound force it back inbound
                    if x_int >= self.image_width:
                        x_int = int(np.floor(x_ori))

                for y in range(self.resized_image_height):
                    # compute the corresponding y position in the original image
                    y_ori = (y / self.resized_image_height) * self.image_height

                    # the amount of interpolation in y axis
                    y_interp = y_ori - np.floor(y_ori)

                    # find the nearest neighbour of current y
                    if y_interp < 0.5:
                        y_int = int(np.floor(y_ori))
                    else:
                        y_int = int(np.ceil(y_ori))
                        # if y_int is out of bound force it back inbound
                        if y_int >= self.image_height:
                            y_int = int(np.floor(y_ori))

                    image_channel_output[x][y] = image_channel[x_int][y_int]

            # insert resized rgb channel image matrix into a list
            image_rgb.append(image_channel_output)
            print("\t\tCompleted")

        # merge 3 image color channel into a image
        image_rgb_output = cv2.merge((image_rgb[0], image_rgb[1], image_rgb[2]))
        print("Completed\n")

        return image_rgb_output         

    # resize image using bilinear interpolation
    def bilinear(self):
        """
            Bilinear Interpoalation
        """
        print("Bilinear Interpolation:")

        # empty list to store each resized rgb channel
        image_rgb = []

        rgb = "bgr"

        B = np.array([[0, 1], [1, 1]])
        B_inv = np.linalg.inv(B) # since B is symmetric matrix B_inv = B_inv_T

        # perform bilinear interpolation for color channel
        for i in range(3):
            print("\tInterpolating {} channel...".format(rgb[i]))
            image_channel = self.image[:, :, i]

            # pre-allcoate space for output image in current color channel
            # make sure the data type is unsigned integer 8 since only uint8 data type are accepted in image file
            image_channel_output = np.zeros((self.resized_image_width, self.resized_image_height), dtype=np.uint8)

            # pad original image to make sure every pixel in the original image can find
            # 4 nearby data points for bilinear interpolation
            # pad row
            bottom_row = image_channel[-1, :]
            image_row_padding = np.vstack((image_channel, bottom_row))

            # pad column
            rightmost_column = image_row_padding[:, -1]
            image_padding = np.c_[image_row_padding, rightmost_column]

            # preallocate space for F matrix
            # F stores f matrix for each pixel in the original image matrix (computed using image after padding)
            # each f is a 2 by 2 matrix that stores 4 pixel values for bilinear interpoolation
            F = np.zeros((self.image_width, self.image_height, 2, 2))

            for x in range(self.image_width):
                for y in range(self.image_height):
                    f = np.array([[image_padding[x][y],   image_padding[x][y+1]],
                                [image_padding[x+1][y], image_padding[x+1][y+1]]])
                    F[x][y] = f

            # interpolate image pixel by pixel
            for x in range(self.resized_image_width):
                # compute the corresponding x position in the original image
                x_ori = (x / self.resized_image_width) * self.image_width

                # the amount of interpolation in x axis(eg: x_ori=1.5, then x_interp=0.5, 0.5 from x=1)
                x_interp = x_ori - np.floor(x_ori)

                # integer part of x_ori (used to index F matrix created above)
                x_int = int(np.floor(x_ori))

                for y in range(self.resized_image_height):
                    # compute the corresponding y position in the original image
                    y_ori = (y / self.resized_image_height) * self.image_height

                    # the amount of interpolation in y axis
                    y_interp = y_ori - np.floor(y_ori)

                    # integer part of y_ori (used to index F matrix created above)
                    y_int = int(np.floor(y_ori))

                    # if the program find that the interpolated point can be found in the original image matrix
                    # eg: (x_ori, y_ori) = (x, y) = (0, 0)
                    # then there is no need to perform bilinear interpolation for that point
                    # we can simply copy the pixel value of that point from original image to scaled image
                    if x_interp==0.0 and y_interp==0.0:
                        # there should always be (orginal image width * original image height) times we can copy pixel value like this
                        # for three rgb channel we can enter into this if statement for 3 * (original image width * origianl image height) times
                        image_channel_output[x][y] = image_channel[int(x_ori)][int(y_ori)]
                    else:
                        # interpolate value in x direction (row vector)
                        X = np.expand_dims(np.array([x_interp**1, x_interp**0]), axis=0)
                        # interpolate value in y direction (column vector)
                        Y = np.expand_dims(np.array([y_interp**1, y_interp**0]), axis=1)

                        # f matrix in this point
                        F_interp = F[x_int][y_int]

                        # bilinear interpolation
                        interpolated_value = X.dot(B_inv).dot(F_interp).dot(B_inv).dot(Y)
                        
                        # after bilinear interpolation between adjacent pixels, floating point value will be returned
                        # instead of usigned integer 8. We need to clamp the value between 0 and 255 in order to show
                        # image matrix as an image
                        if interpolated_value < 0:
                            interpolated_value = 0
                        elif interpolated_value > 255:
                            interpolated_value = 255

                        image_channel_output[x][y] = interpolated_value
            
            # insert resized rgb channel image matrix into a list
            image_rgb.append(image_channel_output)
            print("\t\tCompleted")
        
        # merge 3 image color channel into a image
        image_rgb_output = cv2.merge((image_rgb[0], image_rgb[1], image_rgb[2]))
        print("Completed\n")

        return image_rgb_output


    # resize image using bicubic interpolation
    def bicubic(self):
        """
            Bicubic Interpolation
        """
        print("Bicubic Interpolation:")

        # empty list to store each resized rgb channel
        image_rgb = []

        B = np.array([[-1, 1, -1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [8, 4, 2, 1]])
        B_inv = np.linalg.inv(B)
        B_inv_T = B_inv.T 

        rgb = "bgr"

        # perform bicubic interpolation for each color channel
        for i in range(3):
            print("\tInterpolating {} channel...".format(rgb[i]))
            image_channel = self.image[:, :, i]

            # pre-allcoate space for output image in current color channel
            # make sure the data type is unsigned integer 8 since only uint8 data type are accepted in image file
            image_channel_output = np.zeros((self.resized_image_width, self.resized_image_height), dtype=np.uint8)

            # pad original image channel to make sure every pixel in the original image can find 16 data points to
            # perform bicubic interpolation
            # pad row
            top_row = image_channel[0, :]
            bottom_row = image_channel[-1, :]
            image_row_padding = np.vstack((top_row, image_channel))
            image_row_padding = np.vstack((image_row_padding, bottom_row))
            image_row_padding = np.vstack((image_row_padding, bottom_row))

            # pad column
            leftmost_column = image_row_padding[:, 0]
            rightmost_column = image_row_padding[:, -1]
            image_padding = np.c_[leftmost_column, image_row_padding, rightmost_column, rightmost_column]

            # preallocate space for F matrix
            # F stores f matrix for each pixel in the original image matrix (using image_padding)
            # each f is a 4 by 4 matrix that stores 16 pixel values for bicubic interpolation
            F = np.zeros((self.image_width, self.image_height, 4, 4))

            for x in range(self.image_width):
                x_padding = x + 1 # reposition x index since we are using image martix after padding
                for y in range(self.image_height):
                    y_padding = y + 1 # reposition y index since we are using image matrix after padding
                    
                    # formalize f matrix
                    f = np.array([[image_padding[x_padding-1][y_padding-1], image_padding[x_padding-1][y_padding], image_padding[x_padding-1][y_padding+1], image_padding[x_padding-1][y_padding+2]],
                                  [image_padding[x_padding][y_padding-1],   image_padding[x_padding][y_padding],   image_padding[x_padding][y_padding+1],   image_padding[x_padding][y_padding+2]],
                                  [image_padding[x_padding+1][y_padding-1], image_padding[x_padding+1][y_padding], image_padding[x_padding+1][y_padding+1], image_padding[x_padding+1][y_padding+2]],
                                  [image_padding[x_padding+2][y_padding-1], image_padding[x_padding+2][y_padding], image_padding[x_padding+2][y_padding+1], image_padding[x_padding+2][y_padding+2]]])

                    # store f in F
                    F[x][y] = f

            # interpolate image pixel by pixel
            for x in range(self.resized_image_width):
                # compute the corresponding x position in the original image
                x_ori = (x / self.resized_image_width) * self.image_width

                # the amount of interpolation in x axis(eg: x_ori=1.5, then x_interp=0.5, 0.5 from x=1)
                x_interp = x_ori - np.floor(x_ori)

                # integer part of x_ori (used to index F matrix created above)
                x_int = int(np.floor(x_ori))

                for y in range(self.resized_image_height):
                    # compute the corresponding y position in the original image
                    y_ori = (y / self.resized_image_height) * self.image_height

                    # the amount of interpolation in y axis
                    y_interp = y_ori - np.floor(y_ori)

                    # integer part of y_ori (used to index F matrix created above)
                    y_int = int(np.floor(y_ori))

                    # if the program find that the interpolated point can be found in the original image matrix
                    # eg: (x_ori, y_ori) = (x, y) = (0, 0)
                    # then there is no need to perform bicubic interpolation for that point
                    # we can simply copy the pixel value of that point from original image to scaled image
                    if x_interp==0.0 and y_interp==0.0:
                        # there should always be (orginal image width * original image height) times we can copy pixel value like this
                        # for three rgb channel we can enter into this if statement for 3 * (original image width * origianl image height) times
                        image_channel_output[x][y] = image_channel[int(x_ori)][int(y_ori)]
                    else:
                        # interpolate value in x direction (row vector)
                        X = np.expand_dims(np.array([x_interp**3, x_interp**2, x_interp**1, x_interp**0]), axis=0)
                        # interpolate value in y direction (column vector)
                        Y = np.expand_dims(np.array([y_interp**3, y_interp**2, y_interp**1, y_interp**0]), axis=1)

                        # f matrix at this point
                        F_interp = F[x_int][y_int]

                        # bicubic interpolation
                        interpolated_value = X.dot(B_inv).dot(F_interp).dot(B_inv_T).dot(Y)

                        # after bicubic interpolation between adjacent pixels, floating point value will be returned
                        # instead of usigned integer 8. We need to clamp the value between 0 and 255 in order to show
                        # image matrix as an image
                        if interpolated_value < 0:
                            interpolated_value = 0
                        elif interpolated_value > 255:
                            interpolated_value = 255

                        image_channel_output[x][y] = interpolated_value
            
            # insert resized rgb channel image matrix into a list
            image_rgb.append(image_channel_output)
            print("\t\tCompleted")
        
        # merge 3 image color channel into a image
        image_rgb_output = cv2.merge((image_rgb[0], image_rgb[1], image_rgb[2]))
        print("Completed\n")

        return image_rgb_output