import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Scale, HORIZONTAL

class LineDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hough Transform Line Detection")
        
        self.label = Label(root, text="Select an Image for Line Detection")
        self.label.pack()

        self.select_button = Button(root, text="Choose Image", command=self.select_image)
        self.select_button.pack()
        
        self.canny_scale = Scale(root, from_=50, to=150, orient=HORIZONTAL, label='Canny Threshold')
        self.canny_scale.set(100)
        self.canny_scale.pack()
        
        self.hough_thresh_scale = Scale(root, from_=50, to=200, orient=HORIZONTAL, label='Hough Threshold')
        self.hough_thresh_scale.set(100)
        self.hough_thresh_scale.pack()
        
        self.detect_button = Button(root, text="Detect Lines", command=self.detect_lines)
        self.detect_button.pack()

        self.image_path = None
    
    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        self.label.config(text="Selected image: {}".format(self.image_path))
    
    def detect_lines(self):
        if not self.image_path:
            self.label.config(text='No image selected!')
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        """
        캐니 엣지 검출 - 엣지 검출
        """
        # 최소임계값: 이 값보다 낮은 그라디언트 값은 엣지로 간주되지 않음
        # 최대임계값: 이 값보다 높은 그라디언트 값은 확실한 엣지로 간주
        edges = cv2.Canny(gray, self.canny_scale.get(), self.canny_scale.get()*2)
        
        """
        허프 변환
        - 이미지에서 모양을 찾는 알고리즘
        - 엣지 기반의 직선 검출 알고리즘
        """
        # edges, 1, np.pi/180 ==> 일반적으로 사용되는 값들
        # self.hough_thresh_scale.get(): 특정 지점이 누적된 횟수가 이 값 이상이면 직선으로 간주
        lines = cv2.HoughLines(edges, 1, np.pi/180, self.hough_thresh_scale.get())
        if lines is not None:
            for rho, theta in lines[:,0]: # rho: 직선이 원점에서 떨어진 거리, theta: 직선이 x축과 이루는 각도
                # 허프 변환에서 직선은 극좌표 방정식으로 표현됨 ==> p = x*cos(θ) + y*sin(θ)
                # 극좌표 형태 (rho, theta)를 직선의 두 점 (x1,y1) (x2,y2)로 변환
                a = np.cos(theta)
                b = np.sin(theta)
                
                # 직선 위의 한 점 계산
                x0 = a * rho
                y0 = b * rho
                
                # 1000은 임의로 큰 값으로 설정하여 직선을 충분히 연장해 화면 밖까지 그릴 수 있도록
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imshow('Detected Lines', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    root = Tk()
    app = LineDetectionApp(root)
    root.mainloop()