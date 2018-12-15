# Soccer Offside Detection

## Background
1. One offside decision can change the game result.
2. Can be used by Soccer referee, Soccer live stream, Soccer player & Soccer coach, VAR

## System Diagram
![System Diagram](/material/System_Diagram.png)

## Methods
1. Use OpenCV to abstract images from video
2. Use Hough Line Detection to obtain the boundary 
3. Use color-based threshold to build a mask to identify players
4. Use color-based centroid detection and TLD(Tracking-Learning-Detection) to track the ball
5. Detect several statuses to decide an offside

## Achievements
1. Abstract boundary lines
2. Detect and track players
3. Track the ball
4. Decide when the offside occurs

## Future Steps
1. Incorporate Multiple Field Camera
2. Auto-Adjust Color Thresholds
3. Affine transform with whole pitch view


