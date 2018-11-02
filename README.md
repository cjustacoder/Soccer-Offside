# Soccer-Offside Detection
:soccer:
## Project Abstraction
The goal of this project is to detect an offside situation in a soccer game. 

The program should track the location of each player from both team and the location of the ball. Then an offside line should be drawn which means if the offense team member is beyond this line, an offside situation will occur. Once such situation is detected, the system will give out an alert.

# Approach
1. Detect and track players from each team;
2. Detect the boundary line;
3. Detect the ball;
4. Draw the offside line;
5. Give out alert when offside is detected.

## Player Detection and Tracking
### Method
We use a color based method to detect players from each side. Because each team will wear jersey in different color, and the color can be described in value of HSV(hue, saturation, value). So we use the HSV value to differentiate color, detect and track players.

**P.S.** Green color jersey may not be a problem, since the "green" from a playground and the "green" from jersey may not be exactly the same to a great extent. So these two "green" should have different HSV value, hence we can tell apart these color by setting a precise range of HSV value during we build the filter.

### Implementation
We implement this process by using OpenCV with following steps:

1. Enter the corresponding HSV value of our interest color to find and differentiate players. For example, the range of blue in HSV can be set between ```javascript [110, 50, 50]``` and ```javascript [130, 255, 255]```;
2. Build a mask with these HSV value. The mask will make "holes" in those corresponding areas with HSV value in this range, which can remain these parts from the frame and wipe the other parts out.
![GitHub Logo](/gif/Mask.gif)
3. Use the mask as a filter, to remain parts of the frame with required color, and wipe the rest parts out.
4. Extract boundaries of each remaining parts.
5. Set threshold to exclude those parts with an area under threshold(which are considered to be noises).
6. Draw rectangular around each area remained.
7. Find the center of rectangular and add an offset to find positions of players' feet.
8. Integrate frames into a video and output the video.

## Boundary Line Detection
### Method
Hough transfrom.
### Implementation

# How to Run Our Code
## Environment Configuration
Run with Python 3.6, numpy and opencv-python are needed to be installed.
## Directory Managment
You need to put python script and input video in the same directory.
## Code
The code you need is in the ```javascript src```directory, to test two function mentioned above, you should use file ```javascript track_and_draw.py``` and ```javascript hough_video_lines_detection_pro.py```.
## Material
If you wish to use our material, please go to directory ```javascript material```.

# Demo
To check out accomplishment, you can go to directory ```javascript output_video```.



