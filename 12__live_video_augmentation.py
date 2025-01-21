import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Scale

class VideoAugmentationApp:
    def __init__(self, window):
        self.window = window
        self.window.title('Live Video Augmentation')
        
        self.cap = cv2.VideoCapture('data/person2.mp4')
        self.aug_type = tk.StringVar(value="None")
        
        self.create_widgets()
        self.update()
    
    def create_widgets(self):
        self.video_label = ttk.Label(self.window)
        self.video_label.grid(row=0, column=0, columnspan=4)

        ttk.Label(self.window, text="Augmentation:").grid(row=1, column=0, padx=5, pady=5)
        
        self.aug_selection = ttk.Combobox(self.window, textvariable=self.aug_type, values=["None", "Grayscale", "Sepia", "Sketch"], state="readonly")
        self.aug_selection.grid(row=1, column=1, padx=5, pady=5)

        self.brightness_slider = Scale(self.window, from_=0, to=100, orient='horizontal', label='Brightness')
        self.brightness_slider.set(50)
        self.brightness_slider.grid(row=1, column=2, padx=5, pady=5)
        
        self.quit_button = ttk.Button(self.window, text="Quit", command=self.quit_app)
        self.quit_button.grid(row=1, column=3, padx=5, pady=5)
    
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.apply_augmentation(frame)
            frame = self.adjust_brightness(frame)
            self.display_frame(frame)
        self.window.after(10, self.update)
    
    def apply_augmentation(self, frame):
        aug_type = self.aug_type.get()
        if aug_type == 'Grayscale':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif aug_type == 'Sepia':
            # 행렬 변환을 사용하여 이미지를 세피아 톤으로 변환
            frame = cv2.transform(frame, np.array([[0.272, 0.534, 0.131],
                                                [0.349, 0.686, 0.168],
                                                [0.393, 0.769, 0.189]]))
            # 변환 후의 픽셀 값이 0~255 범위를 벗어나지 않도록 제한
            frame = np.clip(frame, 0, 255)
        elif aug_type == 'Sketch':
            # 그레이스케일 처리
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 흑백 이미지의 밝기 반전
            inv_img = cv2.bitwise_not(gray_img)
            # 블러 처리
            blur_img = cv2.GaussianBlur(inv_img, (21,21), 0)
            # 블러 이미지 합성
            inv_blur_img = cv2.bitwise_not(blur_img)
            # scale(255)를 곱해 1이 되는 값을 255로 변경해 흰색 출력
            frame = cv2.divide(gray_img, inv_blur_img, scale=256.0)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        return frame

    def adjust_brightness(self, frame):
        brightness = self.brightness_slider.get()
        # 이미지의 픽셀 값을 조정하고 스케일링한 후, 절대값을 적용하여 8비트 이미지로 변환하는 함수
        # 이미지 처리 중에 연산을 하다 보면 픽셀 값이 0보다 작거나 255보다 커질 수 있다. 이럴 때 모든 픽셀 값을 안전하게 0에서 255 사이로 변환
        brightness_adjusted = cv2.cv2.convertScaleAbs(frame, alpha=brightness/50)
        
        return brightness_adjusted

    def display_frame(self, frame):
        img = cv2.resize(frame, (640, 480))
        imgtk = cv2.imencode('.png', img)[1].tobytes()
        tkimg = tk.PhotoImage(data=imgtk)
        self.video_label.imgtk = tkimg
        self.video_label.configure(image=tkimg)
    
    def quit_app(self):
        self.cap.release()
        self.window.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = VideoAugmentationApp(root)
    root.mainloop()