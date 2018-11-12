# Spring3

## Ball tracking
1. learn_obj_tracking: using centroid tracking, color based
2. new_object_tracking: using kcf, mediaflow, tld or mil to trackball
3. new_object_tracking_auto: combine No.1 and No.2, can correct tracking when missing. Don't need to draw box using mouse.
## draw offside line
1. RansacVanishingPoint.py: an algorithm made to calculate vanishing point by using lines which parallel in 3d but intersect in 2d.
2. vanishing_point_test.py: using RansacVanishingPoint.py and the lines detected by Canny and Hough, calculate in every frame in the video.
3. draw_offside_line_test.py: use vanishing point and playes' coordinates to decide which player is the last defender, and draw offside line in evry frame.(still need more testing).
