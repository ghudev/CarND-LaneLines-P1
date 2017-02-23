import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
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
    left_lines    = []
    right_lines   = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # discard vertical lines because of undefined slope
            if x2 == x1:
                continue
            slope = (y2-y1)/(x2-x1)
            # discard lines with slopes too high or low as a method of improving the median
            if abs(slope) < 0.4 or abs(slope) > 0.75:
                continue
            if slope > 0:
                right_lines.append([x1, x2, y1, y2])
            else:
                left_lines.append([x1, x2, y1, y2])

    # For Left and Right if Lines are found: Calculate Average Line, Extend from Bottom, Draw on image
    if right_lines:
        right_lane = np.mean(right_lines, axis = 0)
        right_lane = extend_line(right_lane, img.shape[0], img.shape[0]*0.6)
        cv2.line(img, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), color, thickness)
    
    if left_lines:
        left_lane = np.mean(left_lines, axis = 0)
        left_lane = extend_line(left_lane, img.shape[0], img.shape[0]*0.6)
        cv2.line(img, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def color_selected_hsv(img, white_low = np.uint8([20, 0, 200]), white_high = np.uint8([255, 45, 255]), yellow_low = np.uint8([10, 100, 100]), yellow_high = np.uint8([30, 255, 255])):
    """
    Convert Image to HSV and apply a mask for the color ranges including yellow and white
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Create Yellow and White Color Range
    white_range = cv2.inRange(hsv_img, white_low, white_high)
    yellow_range = cv2.inRange(hsv_img, yellow_low, yellow_high)

    # Combine Ranges and Apply as Mask to the Image
    white_or_yellow = cv2.bitwise_or(white_range, yellow_range)
    return cv2.bitwise_and(img, img, mask = white_or_yellow)

def color_selected_hls(img, white_low = np.uint8([0, 200, 0]), white_high = np.uint8([255, 255, 255]), yellow_low = np.uint8([0, 0, 100]), yellow_high = np.uint8([50, 255, 255])):
    """
    Convert Image to HSL and apply a mask for the color ranges including yellow and white
    """
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Create Yellow and White Color Range
    white_range = cv2.inRange(hls_img, white_low, white_high)
    yellow_range = cv2.inRange(hls_img, yellow_low, yellow_high)

    # Combine Ranges and Apply as Mask to the Image
    white_or_yellow = cv2.bitwise_or(white_range, yellow_range)
    return cv2.bitwise_and(img, img, mask = white_or_yellow)

def extend_line(line, bottom, top):
    lx1, lx2, ly1, ly2 = line
    # Calculate Slope and Intercept of the line
    slope = (ly2-ly1)/(lx2-lx1)
    intercept = ly1 - slope * lx1

    # Use calculated Slope and Intercept to calculate x1, x2, y1, y2 Points for a Line extended from the bottom
    x1 = int((bottom - intercept)/slope)
    x2 = int((top - intercept)/slope)
    y1 = int(bottom)
    y2 = int(top)

    return [x1, y1, x2, y2]

def process_image(image):
    # Define all Parameters in a Dictionary. This Dictionary could be changed during runtime, i.e. adapt to changes in Lighting and other variations.
    params = {
        'blur_kernel_size': 3,
        'canny_low' : 50,
        'canny_high' :150,
        'hough_rho' : 1,
        'hough_theta' : np.pi/180,
        'hough_threshold' : 10,
        'hough_min_len' : 20,
        'hough_max_gap' : 200,
        'region_of_interest' : np.int32([np.array([(0, image.shape[0] * 0.9), (image.shape[1] * 0.4, image.shape[0] * 0.6), (image.shape[1] * 0.6, image.shape[0] * 0.6), (image.shape[1], image.shape[0]  * 0.9)])])
    }
    # color selection and grayscaling
    img = color_selected_hls(image)
    fig = plt.figure(figsize=(10,5), dpi=200)
    sp = fig.add_subplot(2,3,1)
    sp.set_title('HSL Color Selection')
    plt.imshow(img)

    img = grayscale(img)
    sp = fig.add_subplot(2,3,2)
    sp.set_title('Grayscaling')
    plt.imshow(img, cmap='gray')
    
    # smoothing image
    img = gaussian_blur(img, params['blur_kernel_size'])
    sp = fig.add_subplot(2,3,3)
    sp.set_title('Gaussian Blur')
    plt.imshow(img, cmap='gray')

    # canny edge detection
    img = canny(img, params['canny_low'], params['canny_high'])
    sp = fig.add_subplot(2,3,4)
    sp.set_title('Canny Edge')
    plt.imshow(img, cmap='gray')
   
    # region masking
    img = region_of_interest(img, params['region_of_interest'])
    sp = fig.add_subplot(2,3,5)
    sp.set_title('Region Masking')
    plt.imshow(img, cmap='gray')
   
    # hough transformation
    img = hough_lines(img, params['hough_rho'], params['hough_theta'], params['hough_threshold'], params['hough_min_len'], params['hough_max_gap'])
    sp = fig.add_subplot(2,3,6)
    sp.set_title('Hough Transform')
    plt.imshow(img, cmap='gray')

    # create result image with overlay
    result = weighted_img(img, image)
    return result


if __name__ == '__main__':
    import os
    for img_name in os.listdir("test_images/"):
        this_image = mpimg.imread("test_images/" + img_name)
        out_img = process_image(this_image)
        mpimg.imsave("output_images/annotated_" + img_name, out_img,format = "jpg")
        plt.savefig('output_images/plot_' + img_name + '.png')

    # Import everything needed to edit/save/watch video clips
    #from moviepy.editor import VideoFileClip
    #from IPython.display import HTML
    
    #white_output = 'output_test_videos/challenge.mp4'
    #clip1 = VideoFileClip("challenge.mp4")
    #white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    #white_clip.write_videofile(white_output, audio=False)

    #white_output = 'output_test_videos/solidYellowLeft.mp4'
    #clip1 = VideoFileClip("solidYellowLeft.mp4")
    #white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    #white_clip.write_videofile(white_output, audio=False)

    #white_output = 'output_test_videos/solidWhiteRight.mp4'
    #clip1 = VideoFileClip("solidWhiteRight.mp4")
    #white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    #white_clip.write_videofile(white_output, audio=False)