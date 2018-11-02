
import cv2
import numpy as np

img = cv2.imread('C:/Users/14455/Desktop/way_offsides.jpg')
# 灰度处理
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# canny边缘处理
edges = cv2.Canny(gray,50,120)
line = 10
minLineLength = 230
maxLineGap = 20
# HoughLinesP函数是概率直线检测，注意区分HoughLines函数
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, lines=line, minLineLength=minLineLength, maxLineGap=maxLineGap)
# 降维处理
lines1 = lines[:,0,:]
# line 函数勾画直线
# (x1,y1),(x2,y2)坐标位置
# (0,255,0)设置BGR通道颜色
# 2 是设置颜色粗浅度
for x1,y1,x2,y2 in lines1:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# 显示图像
cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()
