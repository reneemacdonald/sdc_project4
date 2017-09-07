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

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[4]
        #channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    return lines

def draw_lines(img, lines, color, thickness):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),10)
    

def process_video(clip1):


	while (clip1.isOpened()):
		ret, frame = clip1.read()

		grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = canny(grayscale_image, 50, 150)
		imshape = frame.shape
		regionOfInterest = region_of_interest(edges,np.array([[(1,imshape[0]),(485, 295), (490,295), (imshape[1],imshape[0])]], dtype=np.int32))
		line_img = np.copy(frame)*0
		lines = hough_lines(regionOfInterest, 2, np.pi/180,20,60,90)
		result_of_draw_lines = draw_lines(line_img,lines,(255,0,0),10)
		color_edges = np.dstack((edges, edges, edges))
		line_edges = cv2.addWeighted(color_edges, 0.8, line_img,1,0)       
		cv2.imshow('frame', line_edges)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	clip1.release()
	cv2.destroyAllWindows()

process_video(clip1)


