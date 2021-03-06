# Soccer-Offside Detection
:soccer:
To see our final accomplishment, please go to folder **Final_version**.
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

1. Enter the corresponding HSV value of our interest color to find and differentiate players. For example, the range of blue in HSV can be set between ```[110, 50, 50]``` and ```[130, 255, 255]```;
2. Build a mask with these HSV value. The mask will make "holes" in those corresponding areas with HSV value in this range, which can remain these parts from the frame and wipe the other parts out.
![Mask](/gif/Mask.gif)
3. Use the mask as a filter, to remain parts of the frame with required color, and wipe the rest parts out.
![Res](/gif/Res.gif)
4. Extract boundaries of each remaining parts.
5. Set threshold to exclude those parts with an area under threshold(which are considered to be noises).
6. Draw rectangular around each area remained.
7. Find the center of rectangular and add an offset to find positions of players' feet.
8. Integrate frames into a video and output the video.

## Boundary Line Detection
### Method
Hough transfrom.
### Implementation
we also use the OpenCV sourse to complete the boundary line detection with the following steps:


In this project, we use the Probailistic Hough Line Transform. In this type hough transform, you can see that even for a line with two arguments(maxLineGap and miniLineLength), it takes a lot of computation. Probabilistic Hough Transform is an optimization of Hough Transform we saw. It doesn’t take all the points into consideration, instead take only a random subset of points and that is sufficient for line detection;
1. We need to load the video and take eacg frame;

2. Implement same first two steps with the player tracking module to build the mask;

3. Apply edge detection method on the image by using the Canny detector;

4. Detector the line using the Probailistic Hough Line Transform. In our case, the threshold is 200, the Maxlinegap is 80 and the Minilinelength is 700. The thresholdis the minimum number of intersections to “detect” a line. The  Maxlinegap can allow some gap between points and has higher probability to detect potential line segment. The Minilinelength help us to filter outs some irrelevant line;

5. Use the colors to draw the lines we detect successfully;

# How to Run Our Code
## Environment Configuration
Run with Python 3.6, numpy and opencv-python are needed to be installed.
## Directory Managment
You need to put python script and input video in the same directory.
## Code
The code you need is in the ```src```directory, to test two function mentioned above, you should use file ```track_and_draw.py``` and ```hough_video_lines_detection_pro.py```.
## Material
If you wish to use our material, please go to directory ```material```.

# Demo
To check out accomplishment, you can go to directory ```output_video```.
## Player Detection and Tracking
![Player Tracking](/gif/Player_Track.gif)
## Boundary Lines Detection
![Hough](/gif/Hough.gif)


