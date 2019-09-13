import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


def display_image(arg_img):
    """method to display the image"""
    cv2.imshow('image', arg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def im2double(im):
    """method to get double precision of a channel"""
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max


def channel_svd(channel):
    """
    method to calculate svd of the input data matrix
    :param channel: data matrix whose svd is to be calculatec
    :return: list of three matrices: U, Sigma and V transpose
    """
    [u, sigma, vt] = np.linalg.svd(channel)
    return [u, sigma, vt]



def channel_via_optimal_k(k, u, s_diagonalized, vt):
    """reconstructs a matrix by selecting k signular values"""
    channel_u_k = u[:, :k]
    channel_s_diagonal_k = s_diagonalized[:k, :k]
    channel_vt_k = vt[:k, :]
    print(k)
    print(channel_u_k.shape,channel_s_diagonal_k.shape)
    channel_reconstruction_matrix = np.dot(np.dot(channel_u_k, channel_s_diagonal_k), channel_vt_k)
    channel_reconstruction_matrix = 255 * channel_reconstruction_matrix
    return channel_reconstruction_matrix



img = cv2.imread("F:\\photos\\lord balaji hd wallpapers1.jpg",flags = cv2.IMREAD_COLOR)
#img = img.reshape()
blue_channel = im2double(img[:, :, 0])
green_channel = im2double(img[:, :, 1])
red_channel = im2double(img[:, :, 2])

[u_red, s_red, vt_red] = channel_svd(red_channel)
[u_blue, s_blue, vt_blue] = channel_svd(blue_channel)
[u_green, s_green, vt_green] = channel_svd(green_channel)

rank_channel = np.linalg.matrix_rank(red_channel)


s_red_diagonalize = np.diag(s_red)
s_blue_diagonalize = np.diag(s_blue)
s_green_diagonalize = np.diag(s_green)

# plot images with different number of components
comps = [638, 500, 400, 300, 200, 100]
for i in comps:

	blue_reconstruction_matrix = channel_via_optimal_k(i, u_blue, s_blue_diagonalize, vt_blue)
	green_reconstruction_matrix = channel_via_optimal_k(i, u_green, s_green_diagonalize, vt_green)
	red_reconstruction_matrix = channel_via_optimal_k(i, u_red, s_red_diagonalize, vt_red)

	re_image = cv2.merge((blue_reconstruction_matrix, green_reconstruction_matrix, red_reconstruction_matrix))
	cv2.imwrite('F:\\photos\\restored_image'+str(i)+'.png', re_image)
