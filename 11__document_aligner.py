import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

def rotate_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 흰 영역(240) 이상은 255, 나머지는 검정 (0)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # 이미지 반전. 흰 영역 => 검정 영역 / 검정 영역 => 흰 영역. 결과는 이진화 이미지
    binary = cv2.bitwise_not(binary)
    
    # 이진화 이미지에서 값이 1인 픽셀의 위치를 True로 반환
    coords = np.column_stack(np.where(binary > 0))
    
    if len(coords) == 0:
        return image
    
    # 면적이 가장 작은 직사각형을 계산
    # rect = ((center_x, center_y), (width, height), angle)
    # angle: 기울기를 -90~0도 사이로 반환
    rect = cv2.minAreaRect(coords)
    
    angle = rect[2]
    if angle < 45:
        angle = -angle  # 현재 각도를 그대로 사용하되, 부호를 반대로 하여 수평 정렬을 수행
    else:
        angle = 90 - angle # 90도에서 현재 각도를 빼서 수평 정렬 각도를 계산
    
    (h,w) = image.shape[:2]
    center = (w//2, h//2)
    
    # 회전 변환 행렬 계산
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 이미지 회전
    rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def load_image():
	filepath = filedialog.askopenfilename()
	if not filepath:
		return
	image = cv2.imread(filepath)
	aligned_image = rotate_document(image)
	if aligned_image is not None:
		display_image(aligned_image)
	else:
		messagebox.showerror("Error", "Could not detect document content!")


def display_image(cv_img):
	cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(cv_img)
	img = ImageTk.PhotoImage(img)
	canvas.create_image(0, 0, anchor=tk.NW, image=img)
	canvas.image = img


# Set up the Tkinter interface
app = tk.Tk()
app.title("Document Aligner")
app.geometry("800x900")

canvas = tk.Canvas(app, width=800, height=800)
canvas.pack()

button_frame = tk.Frame(app)
button_frame.pack(fill=tk.X, side=tk.BOTTOM)

load_btn = tk.Button(button_frame, text="Load Image", command=load_image)
load_btn.pack(side=tk.LEFT, padx=10, pady=10)

app.mainloop()