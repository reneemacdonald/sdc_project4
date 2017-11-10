
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: calibration2_undistorted.jpg "Undistorted"
[image2]: warped_lines.png "Warp Example"
[image3]: thresholded_binary_image.png "Thresholded Binary Image"
[image4]: result242.jpg "Lane Area Image"
[image5]: lane_line_pixels.png "Lane Line Pixels"
[image6]: undistorted1.jpg "Undistored Image"
[image7]: distorted1.jpg "Distorted Image"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in lines 327 through 340 of the file called `pipeline.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image7]

Original Image
#####
![alt text][image6]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 61 through 102 in `another_file.py`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 22 through 59 in the file `pipeline.py` (pipeline.py) The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 700      | 350, 700       | 
| 600, 450      | 350, 0      |
| 700, 450     | 950, 0      |
| 1100, 700      | 950, 700        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. If I had had more time I would have worked on a better way to calculate it rather than just putting numbers.

![alt text][image2]

#### 4. In pipeline.py in lines 127 - 299 I identified lane-line pixels using a sliding window search

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In lines 213-215 in pipeline.py I caclulated the offset from the center. In lines 396-398 I calculated the radius of curvature using this forumula R
​curve
​​ =
​∣2A∣
​
​(1+(2Ay+B)
​2
​​ )
​3/2
​​ 
​​

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 350 through 460 in my code in `pipeline.py` in the function `measuring_curvature()`.  I found that my lane area was overflowing the left line a bit so n line 421,I cut off a number of pixels from the left area. On lines 206-208 I checked to see which way the lane was curving and flipped the polynomial if it was curving to the right. Here is an example of my result on a test image:

![alt text][image4]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/Fby-P1Zlwqo)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I found that the s channel worked pretty well but failed in bright light. But the l channel was better but sometimes overestimated the lane line.

If I had more time I would improve it by applying a mask first and only searching for the lines within that area. That would have eliminated some of the problem with the l channel 
vastly overestimating the lines in some areas where it mistook a shadow for the line. Also I would like to break up my one class into several classes and also keep track of the previous ten frames so that if it couldn't find the lines it could use an average of the last ten ones.

