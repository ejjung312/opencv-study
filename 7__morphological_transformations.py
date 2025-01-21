import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import Scale, HORIZONTAL, Button

# Initialize the application
root = tk.Tk()
root.title("Morphological Transformations")

def load_image():
    global img, img_display
    
    file_path = filedialog.askopenfilename()
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is not None:
        apply_transformations()


def apply_transformations(*args):
    global img, img_display
    
    if img is None:
        return
    
    kernel_size = kernel_scale.get()
    operation = var.get()
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'Erosion': # 침식
        transformed_img = cv2.erode(img, kernel, iterations=1)
    elif operation == 'Dilation': # 팽창
        transformed_img = cv2.dilate(img, kernel, iterations=1)
    elif operation == 'Opening': # 침식 후 팽창. 밝은 영역이 줄어들고 어두운 영역이 늘어남
        transformed_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'Closing': # 팽창 후 침식. 어두운 영역이 줄어들고 밝은 영역이 늘어남
        transformed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif operation == 'Gradient': # 그레디언트 = 팽창 - 침식. 그레일스케일 이미지가 가장 급격하게 변하는 곳에서 가장 높은 결과를 반환
        transformed_img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    elif operation == 'Top Hat': # 탑햇 = 입력이미지 - 열림. 밝은 부분 강조
        transformed_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    elif operation == 'Black Hat': # 블랙햇 - 닫힘 - 입력이미지. 어두운 부분 강조
        transformed_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    img_display = transformed_img
    cv2.imshow('Image', img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 드랍다운 메뉴
OPTIONS = ['Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat']
print(*OPTIONS)
var = tk.StringVar(root)
var.set(OPTIONS[0])
operation_menu = tk.OptionMenu(root, var, *OPTIONS)
operation_menu.pack()

# 커널사이즈 조정 슬라이더
kernel_scale = Scale(root, from_=1, to=20, orient=HORIZONTAL, label='Kernel Size')
kernel_scale.set(5)
kernel_scale.pack()

# 이미지 버튼 생성
load_button = Button(root, text="Load Image", command=load_image)
load_button.pack()

# 이미지 업데이트
kernel_scale.bind("<ButtonRelease-1>", lambda x: apply_transformations())
var.trace('w', apply_transformations)

# Start the application
root.mainloop()
cv2.destroyAllWindows()