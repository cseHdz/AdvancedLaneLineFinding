## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project <a name="writeup"></a>
---

The objective of this project is to write a software pipeline to identify the lane boundaries in a video.
The goals / steps are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
1. Apply a distortion correction to raw images.
1. Use color transforms, gradients, etc., to create a thresholded binary image.
1. Apply a perspective transform to rectify binary image ("birds-eye view").
1. Detect lane pixels and fit to find the lane boundary.
1. Determine the curvature of the lane and vehicle position with respect to center.
1. Warp the detected lane boundaries back onto the original image.
1. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames. 

For details on implementation refer to [this file.](./AdvancedLaneFinding.ipynb)

[//]: # (Image References)


[image1]: ./output_images/calibration/calibration_0.jpg "Calibration"
[original1]: ./camera_cal/calibration3.jpg "Calibration"
[image1_2]: ./output_images/calibration/undistorted_chessboard.jpg "Undistorted Chessboard"

[image2]: ./output_images/image_pipeline/undistorted.jpg "Road Transformed"
[image3]: ./output_images/image_pipeline/final_gradient.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/test4.jpg "Output"
[video1]: ./output_videos/project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

1. [Writeup](#writeup) - This file
1. [Camera Calibration](#calibration)
7. [Pipeline (Images)](#images)
7. [Pipeline (Video)](#video)
7. [Discussion](#discussion)

### Camera Calibration <a name="calibration"></a>

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The first step for detecting lanes in an image is to perform camera calibration to identify the matrix and distortion coefficients that describe the camera. To achieve this, we need to map arbitrary points in a camera image to their corresponding points in an undistorted image. For this purpose, we leverage images of chessboards as mapping corners between original and corrected versions is relatively straight forward.

To begin, we prepare "object points", which refer to a 3-dimensional grid created to represent coordinates in the real world. The corrected chessboard image will be assummed to be fixed on the (x,y) plane at z=0, such that object points remain the same across images. Then, iterating through the available calibration images, we proceed to find cheesboard corners through the `cv2.findChessboardCorners()` function. Every time corners are found, the corners coordinates (x,y) will be appended to a list `imgpoints` and the object points to a list `objpoints`.

Once we have iterated through all images we proceed to compute the camera calibration and distortion cousing the `cv2.calibrateCamera()` function. 

Then, we leverage the function `cv2.undistort()` to correct the image.

![alt text][original1]
![alt text][image1]
![alt text][image1_2]


## Pipeline (single images)

2. [Distortion Correction](#distortion)
3. [Tranformations To A Binary Image](#transformations)
4. [Perspective Transform](#perspective)
5. [Detect Line Boundaries](#boundaries)
5. [Lane Curvature & Vehicle Center](#curvature)
5. [Unwarp Image](#unwarp)
7. [Render Lane Lines](#render)

#### 1. Provide an example of a distortion-corrected image. <a name="distortion"></a>

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result. <a name="transformations"></a>

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image. <a name="perspective"></a>

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial? <a name="boundaries"></a>

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center. <a name="curvature"></a>

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly. <a name="render"></a>

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


