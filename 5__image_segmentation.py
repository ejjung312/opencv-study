import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import Button, Label


def select_image():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segment_image(image)


def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    """
    이미지 이진화
    - 임계값을 설정해 임계값보다 크면 흰색, 아니면 검정색
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow('thresh', thresh)
    kernel = np.ones((3,3), np.uint8)
    
    """
    모폴로지 - 이미지를 모양에 대한 정보에 집중
    열기(opening): 침식(erosion) 후 팽창. 노이즈를 제거하는 용도
    닫기(closing): 팽창(dilate) 후 침식
    """
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imshow('opening', opening)
    # 팽창 - 객체 외곽을 확대시키는 연산
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # cv2.imshow('sure_bg', sure_bg)
    
    """
    거리변환 (distance transform)
    - 픽셀값이 0인 배경으로부터의 거리를  픽셀값이 255인 영역에 표현하는 방법
    - 물체 영역의 뼈대를 찾아 영역을 정확히 파악
    """
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # cv2.imshow('dist_transform', dist_transform)
    
    # 0.7 * dist_transform.max() 보다 크면 흰색, 아니면 검정색으로 변경
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # 배경과 전경을 빼서 경계선 영역 정의
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    """
    레이블링
    - 서로 연결되어 있는 객체 픽셀에 고유번호를 할당하여 영역 기반 모양 분석, 레이블맵, 바운딩 박스, 픽셀 개수, 무게 중심 좌표 등을 반환할 수 있게 함
    
    레이블맵
    - 영상의 레이블링을 수행하면 각각의 객체 영역에 고유 번호가 매겨진 2차원 정수 행렬
    https://www.google.com/url?sa=i&url=https%3A%2F%2Flucathree.github.io%2Fpython%2Fday49-2%2F&psig=AOvVaw1Kzx5ExbLyNUlGFz57888n&ust=1737513024280000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCKjW6sbihYsDFQAAAAAdAAAAABAE
    """
    # 레이블맵 생성
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Watershed 알고리즘은 0을 "경계 영역"으로 사용하므로, 레이블 값을 1씩 증가시켜 기존 레이블을 1 이상의 값으로 만듬
    markers = markers + 1
    # 전경과 배경이 확실하지 않은 경계 영역(Unknown 영역)을 0으로 설정
    markers[unknown == 255] = 0
    # 
    markers = cv2.watershed(image, markers)
    # Watershed가 경계선으로 구분한 부분을 시각적으로 표시
    image[markers == -1] = [255,0,0]
    
    display_segmented_image(image)


def display_segmented_image(segmented_image):
    cv2.imshow('Segmented Image', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


app = tk.Tk()
app.title('Image Segmentation Tool')

label = Label(app, text="Select an image to perform segmentation.")
label.pack(pady=10)

select_button = Button(app, text="Select Image", command=select_image)
select_button.pack(pady=10)

app.geometry("300x150")
app.mainloop()