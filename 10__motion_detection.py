import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Detection")
        self.root.geometry("300x150")

        self.start_button = ttk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.status_label = ttk.Label(root, text="Status: Not running")
        self.status_label.pack(pady=5)

        self.running = False
        self.cap = None
    
    def start_detection(self):
        self.running = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_label.configure(text='Status: Running')
        self.detect_motion()
    
    def stop_detection(self):
        self.running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_label.configure(text='Status: Not running')
        
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def detect_motion(self):
        self.cap = cv2.VideoCapture('data/person2.mp4')
        _, prev_frame = self.cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21,21), 0)
        
        while self.running:
            _, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            """ 
            블러 처리
            가우시안 블러 - 전체적인 부드러운 노이즈 제거 및 부드럽게 처리
            """
            gray = cv2.GaussianBlur(gray, (21,21), 0)
            
            # 두 이미지 간 절대 차이 계산
            # 연속된 프레임 간의 차이를 계산해 움직임을 감지
            delta_frame = cv2.absdiff(prev_gray, gray)
            """
            이미지 이진화
            - 임계값을 설정해 임계값보다 크면 흰색, 아니면 검정색
            """
            thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
            # 팽창 - 객체 외곽을 확대시키는 연산
            thresh = cv2.dilate(thresh, None, iterations=2) 
            
            # 외곽선 검출
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 외곽선 감싸는 영역의 면적
                if cv2.contourArea(contour) < 500:
                    continue
                (x,y,w,h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
            cv2.imshow('Motion Detection', frame)
            key = cv2.waitKey(25) & 0xFF
            
            if key == ord('q') or not self.running:
                break
            
            prev_gray = gray.copy()
        
        self.stop_detection()


if __name__ == '__main__':
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.mainloop()