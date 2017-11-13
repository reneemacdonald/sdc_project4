import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os 


clip1= cv2.VideoCapture('project_video.mp4')
image1 = cv2.imread('video_images/original_image559.jpg')

count = 0
c = 0
how_curved = False
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D poitns in image plane
yvalue = 719
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

#height, width, dimensions = result.shape
out = cv2.VideoWriter('output.mp4', fourcc, 20, (1280, 720))
offset_meters = 2.3

previous_warp = mpimg.imread('original_image616.jpg')

def warp(img, original_image):

# to do don't hard code the numbers here 

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
	global yvalue 
	yvalue = img.shape[0]
	##print ("image shape", img.shape[0])
	#rint ("yvalue", yvalue)
	'''
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	ax1.set_title('original')
	ax1.imshow(original_image)
	ax2.set_title('Warped again')
	ax2.imshow(warped)
	plt.show()
	'''
	
	
	find_the_lines(original_image, warped, Minv)

def process_video(clip1):
	


	#print ("inside process video")
	
	#for image in lines_array:
		

	while (clip1.isOpened()):
		ret, frame = clip1.read()

		#h, w = image.shape[:2]
		h, w = frame.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1,(w,h))
		undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
		global c
		c += 1
		#cv2.imwrite("undistorted/original%d.jpg" %c, frame)
		#cv2.imwrite("undistorted/undistorted%d.jpg" %c, undistorted)
	


		luv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2LUV)

		hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)

		rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

		s_channel = hsv[:,:,1]
		l_channel = luv[:,:,0]

		r_channel = rgb[:,:,0]

		v_channel = hsv[:,:,2]

		gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
		abs_sobelx = np.absolute(sobelx)
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

		thresh_min = 20
		thresh_max = 100
		sxbinary = np.zeros_like(scaled_sobel)
		sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)]

		#150 pretty good results
		# not 170 fails in the middle
		l_thresh_min = 150
		l_thresh_max = 255
		l_binary = np.zeros_like(l_channel)
		l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

		r_thresh_min = 225
		r_thresh_max = 255
		r_binary = np.zeros_like(r_channel)
		r_binary[(r_channel >= r_thresh_min) & (r_channel <= r_thresh_max)] = 1

		s_thresh_min = 90
		s_thresh_max = 255
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

		v_thresh_min = 150
		v_thresh_max = 255
		v_binary = np.zeros_like(v_channel)
		v_binary[(v_channel >= v_thresh_min) & (v_channel <= v_thresh_max)] = 1

		color_binary_s = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
		color_binary_l = np.dstack((np.zeros_like(sxbinary), sxbinary, l_binary))

		combined_binary = np.zeros_like(sxbinary)
		combined_binary[(l_binary ==1) & (s_binary == 1)  | (r_binary == 1) ] = 1
		

	#cv2.imshow('frame', scaled_sobel)

	# Plotting thresholded images
		'''
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
		#ax1.set_title('Stacked thresholds')
		#plt.imshow(color_binary)
		ax2.set_title('Combined S channel and gradient thresholds')
		plt.imshow(combined_binary, cmap='gray')
		plt.show()
		'''
		
		warped=warp(combined_binary, undistorted)

		


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
	global count
	count += 1
	midpoint = binary_warped.shape[1]/2
	#print("midpoint",midpoint)
	#midpoint = original_image.shape[1]/2
	#cv2.imwrite("video_images/original_image%d.jpg" %count, original_image)
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	
	#plt.plot(histogram)
	#plt.show()
	#plt.imshow(binary_warped)
	#plt.show()
	
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
	#print ("******last value of leftx", leftx[-1], "count", count)
	#if leftx[-1] > 390:
	#	global how_curved
	#	how_curved = True
	firstx = leftx[0:1]
	#print ("firstx ", firstx)
	lastx = leftx[-1]
	
	#print ("binary warped shape", binary_warped.shape[1])
	camera_position =  (binary_warped.shape[1])/2
	#print ("image_width", image_width)
	rightxlane = rightx[-1]
	#print (firstx, rightxlane)
	lane_center =  (rightxlane + firstx)/2
	#print ("lane width", lane_width)
	offset_pixels = abs(camera_position - lane_center)
	#print ("offset", offset)
	x_m_per_pixel = 3.7/700
	global offset_meters
	offset_meters = x_m_per_pixel * offset_pixels
	#print ("offset_meters", offset_meters)



	# midpoint in image is 640

	#print ("how curved 1", how_curved)	

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
	'''
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	plt.show()
	'''

	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
	                              ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
	                              ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	
	# Shows the region of interests around which we can search
	'''
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	plt.imshow(result)
	plt.show()
	
	
	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	'''

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)curvature
	warp_zero = np.zeros_like(binary_warped)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	# change right fix then bottom  right x is over, plot y then also wrong start up
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts_left]), (255,0, 0))
	cv2.fillPoly(color_warp, np.int_([pts_right]), (0, 0, 255))
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	
	#print ("image shape", img.shape[1])
	new_warp = cv2.warpPerspective(color_warp, Minv, (1280, 720), flags=cv2.INTER_LINEAR) 
	# flipping the image because it goes to the right instead of to the left
	#flipped_image = cv2.flip(new_warp,1)
	#print ("new warp image shape", newwarp.shape)
	#print ("new warp image shape", newwarp_resized.shape)
	# Combine the result with the original image
	#newwarp_resized = newwarp.shape[1::-1]
	#print ("new warp resized", newwarp_resized)
	#print ("how curved", how_curved)
	#if how_curved:
	
	#else:
	#	result = cv2.addWeighted(original_image, 1, flipped_image, 0.3, 0)
	#plt.imshow(original_image)
	#plt.imshow(newwarp)
	#plt.imshow(result)
	#plt.show()

	# Define y-value where we want radius of curvature
	# I'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fitx[0]*y_eval + left_fitx[1])**2)**1.5) / np.absolute(2*left_fitx[0])
	right_curverad = ((1 + (2*right_fitx[0]*y_eval + right_fitx[1])**2)**1.5) / np.absolute(2*right_fitx[0])
	#print(left_curverad, right_curverad)
	
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	#print ("plot y", ploty.shape) #720
	#print ("leftx", leftx.shape) #17228
	#print ("Rightx", rightx.shape) #9975

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
	
	#print ("result shape" ,result.shape)

	#
	

	
	# Calculate the new radii of 
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	curvature = (left_curverad + right_curverad )/2
	
	curvature = (left_curverad + right_curverad )/2

	# for single image debugging

	#result = cv2.addWeighted(original_image, 1, new_warp, 0.3, 0)

	# for video images
	
	global previous_warp
	if firstx < 260 or curvature < 500:
		result = cv2.addWeighted(original_image, 1, previous_warp, 0.3, 0)
	else:
		result = cv2.addWeighted(original_image, 1, new_warp, 0.3, 0)
		previous_warp = new_warp
	
	cv2.putText(result, 'Radius of curvature = %f'% curvature, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	cv2.putText(result, 'Vehicle is = %fm left of center'%offset_meters, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	out.write(result)
		
	#measuring_curvature(original_image, binary_warped, Minv, offset_meters)

def camera_calibration(images):
	# Arrays to store object points and image points from all the images


	# Change thig becauase not 6x8, 9x6 instead
	#print ("image ", image_name)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

	for img, image_name in images:
		img_size = (img.shape[1], img.shape[0])
		i = 0
		
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)
			img_corners = cv2.drawChessboardCorners(img, (9,6), corners, ret)
			
			#i = i + 1
		else:
			print ("No corners found")


		
			
		#plt.imshow(img_corners)
		#plt.show()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
	return mtx, dist
'''
def undistortion(objpoints, imgpoints, img, image_name):
	img_size = (img.shape[1], img.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	name, ext = os.path.splitext(image_name)
	#print ("name ", name)
	#print ("ext ", ext)
	#os.rename(image_name, 'undistorted/' + name + '_undistorted' + ext)
	cv2.imwrite('undistorted/'+name + '_undistorted' + ext, dst)
	
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=30)
	plt.imshow(dst)
	plt.show()
	ax2.set_title('Undistorted Image', fontsize=30)
	
	return ret, mtx, dist, rvecs, tvecs
	'''

def measuring_curvature(original_image, warped, Minv, offset_meters):
	#print (offset_meters)
	# Generate some fake data to represent lane-line pixels
	ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
	quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
	# For each y position generate random x position within +/-50 pix
	# of the line base position in each case (x=200 for left, and x=900 for right)
	leftx = np.array([400 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
	                              for y in ploty])
	rightx = np.array([930 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
	                                for y in ploty])

	leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
	rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
	
	# real lane lines
	'''
	nonzero = warped.nonzero()
	nonzerox = np.array(nonzero[1])
	nonzeroy  = np.array(nonzero[0])
	'''



	# Fit a second order polynomial to pixel positions in each fake lane line
	left_fit = np.polyfit(ploty, leftx, 2)
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fit = np.polyfit(ploty, rightx, 2)
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Plot up the fake data	
	'''
	mark_size = 3
	plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
	plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
	plt.xlim(0, 1280)
	plt.ylim(0, 720)
	plt.plot(left_fitx, ploty, color='green', linewidth=3)
	plt.plot(right_fitx, ploty, color='green', linewidth=3)
	plt.gca().invert_yaxis() # to visualize as we do the images
	plt.show()
	'''

	# Define y-value where we want radius of curvature
	# I'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of 
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	curvature = (left_curverad + right_curverad )/2
	#print("left curved", left_curverad, right_curverad)
	# Now our radius of curvature is in meters
	#print(left_curverad, 'm', right_curverad, 'm')
	# Example values: 632.1 m    626.2 m

	
	#cv2.imwrite("video_images/result%d.jpg" %count, result)



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
# read in image 616 and see where the lines are
camera_calibration(image_array)
# Tues - color/gradient threshold
#undistortion(objpoints, imgpoints, image)
#@def camera_calibration():
'''
'''
lines_array = []
lines_path = 'test_images/*.jpg'
lines_images = glob.glob(lines_path)
for lines_image in lines_images:
	img = cv2.imread(lines_image)
	#print (type(img))
	lines_array.append(img)
	#plt.imshow(img)
	#plt.show()
'''

lines_array = []
lines_array.append(image1)

image_array = []
path ='camera_cal/*.jpg'
images = glob.glob(path)
for image in images:
	img = cv2.imread(image)
	#print ("image", image)
	#plt.imshow(img)
	#plt.show()
	image_array.append((img, image))
mtx, dist = camera_calibration(image_array)


#print (image_array)
'''
lines_array = []
lines_path = 'test_images/straight_lines1.jpg'
lines_images = glob.glob(lines_path)
for lines_image in lines_images:
	img = cv2.imread(lines_image)
	print (type(img))
	lines_array.append(img)
	plt.imshow(img)
	plt.show()
'''
#left_curve = cv2.imread('test_images/test2.jpg')
#process_video(lines_array)


process_video(clip1)

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


# To do:
# apply a mask, only search within a certain area
# Fix color gradients
# Break into separate clases
# Keep track of last ten frames


# get the offset from the center