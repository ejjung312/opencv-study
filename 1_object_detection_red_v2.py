import cv2

cap = cv2.VideoCapture('data/ball_red2.mp4')

while True:
    success, img = cap.read()    
    
    if not success:
        # 동영상 끝에 도달했을 경우 처음으로 되돌림
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # HSV 변환: RGB는 빛의 변화에 민감하기 때문에 HSV(Hue, Saturation, Value)로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. 마스크 생성
    lower_color = (171,93,136)
    upper_color = (179,255,255)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # 3. 컨투어 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    M = {}
    max_i = {}
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            M = cv2.moments(i)
            max_i = i

    # 4. 무게 중심 계산
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.circle(img, (cx,cy), 10, (255,0,0), -1)
    
    # 반지름 계산
    (x,y), radius = cv2.minEnclosingCircle(max_i)
    radius = int(radius)
    
    # 원 둘레 그리기
    cv2.circle(img, (cx,cy), radius, (0,255,0), 3)
    
    cv2.imshow("window", img)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()