# **Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./output_images/annotated_solidWhiteCurve.jpg "Annotated Solid White Curve"
[image2]: ./output_images/plot_solidWhiteCurve.jpg.png "Plot of processing Steps"

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps. In addition all parameters for the steps were stored in a Dictionary.
All steps after the first were performed on the ouput image of the previous step.

In order to more accurately detect yellow lines in the challenging example a processing before grayscaling was needed, because after grayscaling the yellow line wasn't always detectable.
Step 1 was therefore to mask the image by color range. The way that was most effective for me was to convert the image into HSL or HSV colorspace and define the color ranges in these color spaces.
HSL Colorspace (in OpenCV HLS) allowed an easier distinction, especially for the white color, and was therefore chosen for step 1.

The following steps consisted of the steps used in the examples:

Step 2 changed the image to a grayscale image. Step 3 applied a gaussian blur with a Kernel Size of 3. 

Step 4 used Canny Edge detection with a low threshold of 50 and high threshold of 150. The Parameters were chosen after various tests with different values.

Step 5 applied a region of interest. In order to reduce lines found in the reflection of the hood of the car in the more challenging video the bottom of the region was set to 90% of the image. 
The top of the region was chosen to be 40% of the image from the left and right as this was sufficient to detect lines even in curves.

Step 6 was the line detection through hough transformation with a modified draw_lines function.
In order to seperate lines between the left and right a slope was calculated and lines with a slope bigger than 0 were part of the right lane lines and lines with a slope lower than 0 part of the left.
In addition lines with slopes too high or too low were discarded as a preparation to calulate the average for each side. The range for valid slopes was chosen based on plottings of found slopes during test runs and most slopes had an absolute value between 0.5 and 0.75.
After seperating all lines into left and right an average was calculated for both left and right. The average lines for left and right where then extended from the bottom of the image to a point at the upper edge of the region of interest using slope and intercept of these lines.
The extended Lines where then drawn onto the image.

These steps can be seen in the Plot.

![Plot with steps 1-6][image2]

The last step consisted of creating the result image with the extended lines as an overlay through the weighted_img function as seen in the next image.

![Final Image][image1]

### 2. Identify potential shortcomings with your current pipeline

One Shortcoming is currently the hardwired Parameters. Under strong lighting changes the color masking might produce large slabs of white which then would not allow for detection of the actual lines.

Lines are either detected or not detected on a per image base. If lines are not detected in an image on one of the sides the markings fail completely, even if they were detected just one frame before.

Curves in the Lane Lines are not detected as such. Currently only a straight lane line is assumed to exist which would not be sufficient in stronger curves.

I have not yet profiled the pipeline too see potential inefficient parts.

### 3. Suggest possible improvements to your pipeline

One Improvement would be to adjust parameters for following processing steps. For example the canny edge thresholds could be adjusted based on a gradient of the image before running the canny edge detection.
Similar steps could be used before other processing steps like with the hough transformation thresholds.
This could be implemented by changing the parameter Dictionary with additional steps.

Lane lines should be improved to possibly include curves, for example by segmenting them.

The color ranges used in color selection steps could be improved. By looking at the HSV and HSL values for white and yellow markings over a variety of test images, better ranges could be found.

Efficiency might be improved by profiling the Code and adjusting it in slower parts.

A more modular approach to the pipeline would be better and allow for defining the steps by configuration.