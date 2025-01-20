import cv2
import numpy as np

# cap = cv2.VideoCapture('data/person1.mp4')
cap = cv2.VideoCapture('data/person2.mp4')
# cap = cv2.VideoCapture('data/person3.mp4')

# cv2.imshow('window', img)
# cv2.imshow('edges', edges)
# cv2.imshow('cartoon', cartoon)
# cv2.waitKey(0)

while True:
    success, img = cap.read()
    
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    ### 엣지 검출
    # 그레이스케일 변환 - 엣지 검출 전 색상 제거
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """ 히스토그램 평활화 """
    # 이미지 밝기 대비 개선
    # gray = cv2.equalizeHist(gray)

    # CLAHE - (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    """ 블러 처리 """
    # 가우시안 블러 - 전체적인 부드러운 노이즈 제거 및 부드럽게 처리
    # blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # 메디안 블러 - 엣지를 더 보존하면서 노이즈 제거
    blurred = cv2.medianBlur(gray, 5)

    """ 엣지 검출 """
    # 캐니 엣지 검출 - 엣지 검출
    # 최소임계값: 이 값보다 낮은 그라디언트 값은 엣지로 간주되지 않음
    # 최대임계값: 이 값보다 높은 그라디언트 값은 확실한 엣지로 간주
    # edges = cv2.Canny(blurred, 100, 200)

    # 자동 임계값 계산
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)

    """ 엣지 연결 - 모폴로지 연산을 사용하여 연결 """
    # 사각형 커널 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # 팽창 - 커널을 적용하여 하나라도 1이 있으면 중심 픽셀을 1로 만듬
    dilated = cv2.dilate(edges, kernel, iterations=1)
    # 침식 - 커널을 적용하여 하나라도 0이 있으면 중심 픽셀 제거
    edges = cv2.erode(dilated, kernel, iterations=1)

    """ 필터(가우시안, 양방향 등) """
    # 색상 단순화 - 색상 평탄화. 잡음 제거
    # cv2.bilateralFilter: 양방향 필터링 함수. 엣지가 아닌 부분만 블러링하여 에지 보존
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # color = cv2.bilateralFilter(img, d=-1, sigmaColor=75, sigmaSpace=75)

    # 엣지와 색상 결합
    edges = cv2.bitwise_not(edges)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 색상과 윤곽선 결합
    cartoon = cv2.bitwise_and(color, edges_colored)

    cv2.imshow('cartoon', cartoon)
    cv2.imshow('edges', edges)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()