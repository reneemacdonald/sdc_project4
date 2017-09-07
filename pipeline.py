import numpy as np
import cv2


clip1= cv2.VideoCapture('project_video.mp4')


def process_video(clip1):


	while (clip1.isOpened()):
		ret, frame = clip1.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame', gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	clip1.release()
	cv2.destroyAllWindows()

process_video(clip1)


