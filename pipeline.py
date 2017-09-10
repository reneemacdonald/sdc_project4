import numpy as np
import cv2
import matplotlib.pyplot as plt



clip1= cv2.VideoCapture('project_video.mp4')

def warp(img):
	img_size = (img.shape[1], img.shape[0])

	src = np.float32(
		[[850, 320],
		[865, 450],
		[533, 250],
		[535, 210]])

	dst = np.float32(
		[[870, 240],
		[870, 370],
		[520, 370],
		[520, 240]])

	M = cv2.getPerspectiveTransform(src, dst)

	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    #return warped

def process_video(clip1):


	while (clip1.isOpened()):
		ret, frame = clip1.read()

		hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
		s_channel = hls[:,:,2]

		frame = np.copy(frame)

		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
		abs_sobelx = np.absolute(sobelx)
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

		thresh_min = 20
		thresh_max = 100
		sxbinary = np.zeros_like(scaled_sobel)
		sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)]

		s_thresh_min = 170
		s_thresh_max = 255
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

		color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

		combined_binary = np.zeros_like(sxbinary)
		combined_binary[(s_binary ==1) | (sxbinary == 1)] = 1

		
		cv2.imshow('frame', scaled_sobel)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		#return combined_binary

	clip1.release()
	cv2.destroyAllWindows()

color_and_gradient = process_video(clip1)