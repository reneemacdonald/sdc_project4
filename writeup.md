
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

The code for this step is in lines 408 through 439 of the file called `pipeline.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used a chessboard to calculate the distortion because it is very easy to see what a chessboard normally looks like vs how it looks like in the images. And it's easy to 
detect automatically because of its pattern. Therefore
making it easy to map distorted points to undistorted points. I then used the values I got to undistort my images before searching for the lane lines:
![alt text][image7]

Original Image
#####
![alt text][image6]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 89 through 120 in `pipeline.py`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 26 through 64 in the file `pipeline.py` (pipeline.py) The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I hardcoded the source and destination points in the following manner:

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

A better way would have been to dynamically generate the points, because these points would only work on similar roads. For instance on roads that are much narrower these points would fail. 

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 700      | 350, 700       | 
| 600, 450      | 350, 0      |
| 700, 450     | 950, 0      |
| 1100, 700      | 950, 700        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. If I had had more time I would have worked on a better way to calculate it rather than just putting numbers.

![alt text][image2]

#### 4. In pipeline.py in lines 127 - 299 I identified lane-line pixels using a sliding window search. First I plotted a histogram of the pixel values along each column in teh image. 
In the histogram I used the two most prominent peaks as a good indication of the x position of the base of the lane lines. I began searching for the lane lines there. I divided the image into 9 areas to search for the lines. But this method failed when there were a lot of shadows and the pixels values for the shadow were very high. In that case, it drew the line almost to the edge of barrier. To correct for that issue if I had more time I would just search around the area where the lane lines were previously found and that would have acted as a kind of mask. But instead I did a validation to check whether whether where it found the first x pixel was realistic or not. I noticed most times it was around 400 but in some cases it was 50 or 250 or even 0. So in my validation I used the previous frame where the firstx was less than 250. However, that isn't such a good approach because with other images it's possible that the average for where the firstx is might have been lower such as 200 then excluding anything under 250 would have meant there wasn't a first good frame or else most of the lane areas would have been the same. I could have skipped the sliding windows search if I had previously found a fit. But when I tried doing that, that part wasn't working for me so I just went back to finding it every time. But if you search every time then it is slower and also it means you need to do a validation if the first x values is very far off. If I had just searching around the general area where the lines would found before then I might not have needed to do a validation for the first x. I also tried searching with 4 lane windows and that performed almost as well as dividing the frame into 9 windows. But I believe 9 windows is probably better because it's more general. In our case, the lines were either straight or only slightly curved there were no sharp curves. I believe that if there were sharp curves then 4 windows wouldn't be enough. Also I might have gotten better results if I had decreased the number of pixels need to be found to recenter the window. As in the sample image I've included, the last window on the right lane doesn't contain the line. And in even the second to last window, some of the line is outside the window.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In lines 248-252 in pipeline.py I caclulated the offset from the center. In lines 389-393 I calculated the radius of curvature using this forumula R
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

I implemented this step in lines 461 through 521 in my code in `pipeline.py` in the function `measuring_curvature()`. Here is an example of my result on a test image:

![alt text][image4]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/uq0bNvJNvWg)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline would probably fail in poor weather conditions such as if there were snow or rain which would hamper visibility. It might also fail at night as all the situations I've tested it on are during the day. 

If I had more time I would improve it by keeping track of the past ten good frames and then if the lines couldn't be detetected or failed the validation criteria, it would use an average of those past ten frames. Also I would change the validation to not check the starting x value but rather the width of the lane or whether the lines were parallel as that could be used on other videos as well.  However, by using the l from luv combined with the s from hsv or the r from rgb, I found that it accurately finds the lane lines and there isn't a space or overlap as there was before. Currently when it fails validation, I take the previous frame. However, I could just keep the previous measurements. Also I would like to break up my one file into several clases and not use global variables. I would also like to detect if it previously found the lane and if so search around a general area. 
