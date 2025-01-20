import cv2
import numpy as np

"""
Hue (H): 색상, 0~179 (빨강=0, 초록=60, 파랑=120)
Saturation (S): 색의 강도, 0~255 (0은 회색, 255는 강한 색)
Value (V): 밝기, 0~255 (0은 어둡고, 255는 밝음)
"""

def nothing(x):
    pass

# 트랙바 생성
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

# cap = cv2.VideoCapture('data/ball_red.mp4')
cap = cv2.VideoCapture('data/ball_red2.mp4')

while True:
    success, img = cap.read()    
    if not success:
        # 동영상 끝에 도달했을 경우 처음으로 되돌림
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # HSV 변환: RGB는 빛의 변화에 민감하기 때문에 HSV(Hue, Saturation, Value)로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 트랙바 값 읽기
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    # 마스크 생성
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(img, img, mask=mask)


    cv2.imshow("img", img)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()