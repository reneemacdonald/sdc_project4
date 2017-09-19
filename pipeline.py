import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os 


#clip1= cv2.VideoCapture('project_video.mp4')
image1 = mpimg.imread('test_images/straight_lines1.jpg')


objpoints = [] # 3D points in real world space
imgpoints = [] # 2D poitns in image plane

def warp(img, original_image):

		
	img_size = (img.shape[1], img.shape[0])
	
	src = np.float32(
		[[210, 700],
		[600, 450],
		[700, 450],
		[1100, 700]])

	dst = np.float32(
		[[350, 700],
		[350, 0], 
		[950, 0],
		[950,  700]])

    

	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	ax1.set_title('original')
	ax1.imshow(original_image)
	ax2.set_title('Warped again')
	ax2.imshow(warped)
	plt.show()
	find_the_lines(original_image, warped, Minv)

def process_video(lines_array):

	print ("inside process video")
	
	for image in lines_array:
		

	#while (clip1.isOpened()):
	#	ret, frame = clip1.read()

		hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
		s_channel = hls[:,:,2]

		frame = np.copy(image)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

	#cv2.imshow('frame', scaled_sobel)

	# Plotting thresholded images
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
		ax1.set_title('Stacked thresholds')
		ax1.imshow(color_binary)


		ax2.set_title('Combined S channel and gradient thresholds')
		ax2.imshow(combined_binary, cmap='gray')
		plt.show()
		warped=warp(combined_binary, image)
		


	#find_the_lines(scaled_sobel)

		
	
	#	if cv2.waitKey(1) & 0xFF == ord('q'):
	#		break

		#return scaled_sobel

	#clip1.release()
	#cv2.destroyAllWindows()

def find_the_lines(original_image, binary_warped, Minv):
	#print (binary_warped.shape)
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Crxeate an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
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
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
		(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
		(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	#	 Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)



	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	#scv2.imshow('frame', out_img)
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	plt.show()
	measuring_curvature(original_image, binary_warped, Minv)

def camera_calibration(images):
	# Arrays to store object points and image points from all the images


	# Change thig becauase not 6x8, 9x6 instead
	#print ("image ", image_name)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates
	i = 0
	for img, image_name in images:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)
			
			img_corners = cv2.drawChessboardCorners(img, (9,6), corners, ret)
			
			plt.imshow(img_corners)
			plt.show()
			undistortion(objpoints, imgpoints, img_corners, image_name)
			i = i + 1
		else:
			print ("No corners found")

def undistortion(objpoints, imgpoints, img, image_name):
	img_size = (img.shape[1], img.shape[0])

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

	dst = cv2.undistort(img, mtx, dist, None, mtx)


	name, ext = os.path.splitext(image_name)
	#print ("name ", name)
	#print ("ext ", ext)

	#os.rename(image_name, 'undistorted/' + name + '_undistorted' + ext)
	cv2.imwrite('undistorted/'+name + '_undistorted' + ext, dst)

	#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	#ax1.imshow(img)
	#ax1.set_title('Original Image', fontsize=30)
	plt.imshow(dst)
	plt.show()
	#ax2.set_title('Undistorted Image', fontsize=30)

def measuring_curvature(original_image, warped, Minv):
	# Generate some fake data to represent lane-line pixels
	ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
	quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
	# For each y position generate random x position within +/-50 pix
	# of the line base position in each case (x=200 for left, and x=900 for right)
	leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
	                              for y in ploty])
	rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
	                                for y in ploty])

	leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
	rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


	# Fit a second order polynomial to pixel positions in each fake lane line
	left_fit = np.polyfit(ploty, leftx, 2)
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fit = np.polyfit(ploty, rightx, 2)
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Plot up the fake data	
	mark_size = 3
	plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
	plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
	plt.xlim(0, 1280)
	plt.ylim(0, 720)
	plt.plot(left_fitx, ploty, color='green', linewidth=3)
	plt.plot(right_fitx, ploty, color='green', linewidth=3)
	plt.gca().invert_yaxis() # to visualize as we do the images
	plt.show()

	# Define y-value where we want radius of curvature
	# I'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	print("left curved", left_curverad, right_curverad)

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	warp_zero = np.zeros_like(warped)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	
	#print ("image shape", img.shape[1])
	newwarp = cv2.warpPerspective(color_warp, Minv, (1280, 720), flags=cv2.INTER_LINEAR) 
	print ("new warp image shape", newwarp.shape)
	#print ("new warp image shape", newwarp_resized.shape)
	# Combine the result with the original image
	#newwarp_resized = newwarp.shape[1::-1]
	#print ("new warp resized", newwarp_resized)
	result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
	plt.imshow(original_image)
	#plt.imshow(newwarp)
	plt.imshow(result)
	plt.show()


# Example values: 1926.74 1908.48

#commenting out because testing out other areas
'''
image_array = []
path ='camera_cal/*.jpg'
images = glob.glob(path)
for image in images:
	img = cv2.imread(image)
	#print ("image", image)
	#plt.imshow(img)
	#plt.show()
	image_array.append((img, image))
print (image_array)

camera_calibration(image_array)

# Tues - color/gradient threshold


#undistortion(objpoints, imgpoints, image)

#@def camera_calibration():

'''

lines_array = []
lines_path = 'test_images/*.jpg'
lines_images = glob.glob(lines_path)
for lines_image in lines_images:
	img = cv2.imread(lines_image)
	print (type(img))
	lines_array.append(img)
	#plt.imshow(img)
	#plt.show()

process_video(lines_array)

#find_the_lines(color_and_gradient)
#exit()
'''
warped_im = warp(image1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Source Image')
ax1.imshow(image1)

ax2.set_title('Warped Image')
plt.imshow(warped_im)
plt.show()

#warp(image1)

'''


# To do work on teh warp image next - perspective transform thurs then fri work on lane lines


# Fri chec that it's working on curved lines

# work on detect lane lines

# sat work on lane curvatue



