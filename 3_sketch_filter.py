import cv2
import numpy as np

# cap = cv2.VideoCapture('data/person1.mp4')
# cap = cv2.VideoCapture('data/person2.mp4')
cap = cv2.VideoCapture('data/person3.mp4')


# cv2.imshow('img', img)
# cv2.imshow('sketch', sketch)
# cv2.waitKey(0)

while True:
    success, img = cap.read()
    
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    """그레이스케일 처리"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """색상 반전"""
    # 흑백 이미지의 밝기 반전
    inverted = cv2.bitwise_not(gray)

    """블러 처리"""
    # 커널 크기가 작아지면 세밀한 선, 크게 하면 부드러운 선
    blurred = cv2.GaussianBlur(inverted, (21,21), 0)

    """블러 이미지 합성"""
    inverted_blurred = cv2.bitwise_not(blurred)

    """스케치 필터 적용"""
    # scale(255)를 곱해 1이 되는 값을 255로 변경해 흰색 출력
    sketch = cv2.divide(gray, inverted_blurred, scale=256)

    cv2.imshow('img', img)
    cv2.imshow('sketch', sketch)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()