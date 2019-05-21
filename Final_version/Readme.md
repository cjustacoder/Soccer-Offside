# Soccer Offside Detection

## Background
1. One offside decision can change the game result.
2. Can be used by Soccer referee, Soccer live stream, Soccer player & Soccer coach, VAR

## System Diagram
![System Diagram](/Final_version/material/System.png)

## Methods
1. Use OpenCV to abstract images from video
![Original](/Final_version/material/Picture1.jpg)
2. Use Hough Line Detection to obtain the boundary 
![Hough](/Final_version/material/Picture2.png)
3. Use color-based threshold to build a mask to identify players
![Player](/Final_version/material/Picture3.png)
4. Use color-based centroid detection and TLD(Tracking-Learning-Detection) to track the ball
![Ball](/Final_version/material/Picture4.png)
5. Detect several statuses to decide an offside
![Algorithm](/Final_version/material/Picture5.png)

## Achievements
1. Abstract boundary lines
2. Detect and track players
3. Track the ball
4. Decide when the offside occurs
![Result](/Final_version/material/Picture6.png)

## Future Steps
1. Incorporate Multiple Field Camera
2. Auto-Adjust Color Thresholds by background subtraction and foreground statistics
3. Affine transform with whole pitch view

## Demo
To review our demo, please look inside folder **outputvideo**

