import numpy as np
import cv2


clip1= cv2.VideoCapture('project_video.mp4')

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscale conversion
    return gray

def gaussian_noise(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""


    # Define a kernel size for Gaussian smoothing / blurring
    kernel_size = 5
    blur_gray = gaussian_noise(img, kernel_size)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    return cv2.Canny(img, low_threshold, high_threshold)


def process_video(clip1):


	while (clip1.isOpened()):
		ret, frame = clip1.read()

		grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = canny(grayscale_image, 50, 150)

		cv2.imshow('frame', edges)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	clip1.release()
	cv2.destroyAllWindows()

process_video(clip1)


