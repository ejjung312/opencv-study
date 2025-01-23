import cv2
import numpy as np
import time

# cap = cv2.VideoCapture('data/person5.mp4')
cap = cv2.VideoCapture('data/person6.mp4')

time.sleep(3)
background = 0
# ret, frame = cap.read()
# background = np.zeros_like(frame)

# 2초에 나왔을 떄 배경이 비교적 잘 나오기 때문에 50번째 프레임으로 설정
# 프레임번호 = 초 * 프레임 비율
ret, background = cap.read() # person6.mp4은 빨간색이 없는 첫번째 프레임 써야함
# for i in range(50):
#     ret, background = cap.read()

# 배열 뒤집기
# background = np.flip(background, axis=1)

# cap.read 초기화
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    success, img = cap.read()
    
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # ??
    # img = np.flip(img, axis=1)
    
    # BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    """
    H-색조
    빨간색: 0~60
    노란색: 61~120
    녹색: 121~180
    시안(cyan): 181~240
    파란색: 241~300
    마젠타(magenta): 301~360
    - 빨강(0°), 초록(120°), 파랑(240°)
    
    S-채도
    - 색상의 회색양
    - 0: 회색, 255: 선명
    
    V-값
    - 색상의 밝기나 강도
    - 0: 어두움, 255: 밝음
    """
    
    # 빨간색 감지를 위한 마스크 생성
    # lower_red ~ upper_red 내의 색상을 감지
    lower_red = np.array([0,120,50])
    upper_red = np.array([10,255,255])
    # lower_red = np.array([0,100,30])
    # upper_red = np.array([15,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # lower_red = np.array([160,100,30])
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    # 빨간색 첫 번째 범위와 두 번째 범위에 속하는 모든 픽셀을 하나의 마스크로 합침
    mask1 = mask1 + mask2
    
    ###
    # 빨간색 마스크에서 외곽선 감지
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 빈 마스크 생성
    filled_mask = np.zeros_like(mask1)
    
    # 외곽선 내부 채우기
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # 내부 영역 채우기
    # finalOutput[filled_mask == 255] = [0,0,255]
    ###
    
    # 모폴로지
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    
    mask2 = cv2.bitwise_not(mask1)
    
    res1 = cv2.bitwise_and(img, img, mask=mask2)
    res2 = cv2.bitwise_and(background, background, mask=mask1)
    
    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
    
    # 빨간색이 있는지 확인
    # red_pixel_count = cv2.countNonZero(mask1)
    # print(red_pixel_count)
    
    cv2.imshow('magic', finalOutput)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()