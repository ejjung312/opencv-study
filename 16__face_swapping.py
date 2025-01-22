import cv2
import numpy as np
import dlib
import threading
from tkinter import filedialog, messagebox, ttk
from tkinter import *
from PIL import Image, ImageTk

# face detector, landmark predictor 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image_path_a = ""
image_path_b = ""
panelA = None
panelB = None
panelC = None
root = None
progress_bar = None
is_processing = False

def select_image(is_first_image=True):
    global panelA, panelB, image_path_a, image_path_b
    
    image_path = filedialog.askopenfilename()
    if len(image_path) > 0:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if is_first_image:
            image_path_a = image_path
            if panelA is None:
                panelA = Label(image=image)
                panelA.image = image
                panelA.pack(side="left", padx=10, pady=10)
            else:
                panelA.configure(image=image)
                panelA.image = image
        else:
            image_path_b = image_path
            if panelB is None:
                panelB = Label(image=image)
                panelB.image = image
                panelB.pack(side="right", padx=10, pady=10)
            else:
                panelB.configure(image=image)
                panelB.image = image


def save_image():
    if panelC is None or not hasattr(panelC, 'image'):
        messagebox.showerror("Error", "No image to save. Please swap faces first.")
        return
    
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    if file_path:
        image = ImageTk.getimage(panelC.image)
        image.save(file_path)
        messagebox.showinfo("Success", "Image saved successfully to {}".format(file_path))


def swap_faces_thread():
    global is_processing
    is_processing = True
    progress_bar.start(10)
    swap_faces()
    progress_bar.stop()
    is_processing = False


def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    # 감지된 얼굴 중 첫번째 얼굴(faces[0])의 포인트 리스트반환(.parts()) 후 x,y(p.x, p.y)로 추출해 matrix 객체로 변환
    return np.matrix([[p.x, p.y] for p in predictor(image, faces[0]).parts()])


def transformation_from_points(points1, points2):
    """
    두 집합의 점(points1, points2) 사이의 유사 변환을 계산하는 알고리즘
    유사변환: 점의 크기, 회전, 이동을 포함하여 두 점 집한 간의 정렬 수행
    points1의 점들을 변환하여 points2와 일치
    """
    # 정밀 계산을 위해 부동소수점 형식으로 변환
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    # 점 집합의 중심 계산
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    # 각 점 집합의 중심을 원점(0,0)으로 이동
    points1 -= c1
    points2 -= c2
    # 점 집합의 표준편차 계산. 스케일(크기) 비율
    s1 = np.std(points1)
    s2 = np.std(points2)
    # 점 집합을 표준 편차로 나누어 스케일을 1로 표준화. 크기의 영향을 제거하고 회전 및 이동만 남김
    points1 /= s1
    points2 /= s2
    
    # 특이값 분해(SVD)
    # points1.T * points2 ==> 두 점 집합 간의 공분산 행렬
    # u: 좌측 직교 행렬. 스케일링 데이터를 최종적으로 회전, 결과 데이터 얻음
    # s: 특이값. 새로운 좌표계에서 데이터 스케일링(중요도)
    # vt: 우측 직교 행렬. 원래 데이터를 직교 변환하여 새로운 좌표계 생성
    u, s, vt = np.linalg.svd(points1.T * points2)
    
    # 회전 행렬 계산. points1을 points2에 맞추는 회전
    r = (u*vt).T
    
    # ((s2/s1)*r ==> 스케일과 회전을 결합한 행렬
    # c2.T-(s2/s1)*r*c1.T ==> 이동 벡터 계산
    # transform은 유사 변환 행렬로 크기,회전,이동이 모두 포함되어 있음
    transform = np.vstack([np.hstack(((s2/s1)*r, c2.T-(s2/s1)*r*c1.T)), np.matrix([0., 0., 1.])])
    
    return transform


def warp_image(image, transform, shape):
    # 이미지 변환
    warped = cv2.warpAffine(image, transform[:2], (shape[1], shape[0]), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    return warped


def correct_colors(im1, im2, landmarks1):
    """
    이미지 결합
    """
    # landmarks1[36:42] => 왼쪽 눈 / landmarks1[42:48] => 오른쪽 눈
    # 왼쪽눈과 오른쪽 눈의 중심 좌표 계산 후 유클리드 거리(직선거리) 계산.
    # 0.4로 조정하여 블러 크기 결정
    blur_amount = 0.4 * np.linalg.norm(np.mean(landmarks1[36:42], axis=0) - np.mean(landmarks1[42:48], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    
    # 원본이미지(im1)에서 블러 처리된 이미지(im1_blur)를 빼서 디테일한 정보인 날카로운 부분이나 텍스처 추출
    # im2에 디테일한 부분을 더해 새로운 이미지 생성
    # 0~255 사이로 제한하기 위해 np.clip 실행
    return np.clip(im2.astype(np.float32)+im1.astype(np.float32) - im1_blur.astype(np.float32), 0, 255).astype(np.uint8)


def swap_faces():
    global panelC
    
    if not image_path_a or not image_path_b:
        messagebox.showerror("Error", "Please select both images first.")
        return
    
    image_a = cv2.imread(image_path_a)
    image_b = cv2.imread(image_path_b)
    
    landmarks1 = get_landmarks(image_a)
    landmarks2 = get_landmarks(image_b)
    
    if landmarks1 is None or landmarks2 is None:
        messagebox.showerror("Error", "Could not detect faces. Make sure the detector model is available and faces are clearly visible.")
        return
    
    # 랜드마크를 기준으로 바운딩 박스 정보
    x1,y1,w1,h1 = cv2.boundingRect(np.array(landmarks1))
    x2,y2,w2,h2 = cv2.boundingRect(np.array(landmarks2))
    
    # 바운딩 박스의 중심점
    center1 = (x1+w1//2, y1+h1//2)
    center2 = (x2+w2//2, y2+h2//2)
    
    # 첫번째 얼굴
    transform = transformation_from_points(landmarks1, landmarks2)
    warped_mask = warp_image(image_b, transform, image_a.shape)
    correct_image = correct_colors(image_a, warped_mask, landmarks1)
    
    # 마스크 생성
    mask = np.zeros(image_a.shape, dtype=image_a.dtype)
    # cv2.convexHull => 얼굴 외곽선을 포함하는 다각형 생성
    # cv2.fillConvexPoly(mask ...) => mask에 다각형 영역을 흰색으로 채움
    cv2.fillConvexPoly(mask, cv2.convexHull(landmarks1), (255,255,255))
    
    # 위치 조정
    # 포아송 이미지 편집 기술을 사용해 이미지를 합성하는 기능
    final_output1 = cv2.seamlessClone(correct_image, image_a, mask, center1, cv2.NORMAL_CLONE)
    
    # 두번째 얼굴
    transform = transformation_from_points(landmarks2, landmarks1)
    warped_mask = warp_image(image_a, transform, image_b.shape)
    correct_image = correct_colors(image_b, warped_mask, landmarks2)
    
    mask = np.zeros(image_b.shape, dtype=image_b.dtype)
    cv2.fillConvexPoly(mask, cv2.convexHull(landmarks2), (255, 255, 255))
    
    final_output2 = cv2.seamlessClone(correct_image, image_b, mask, center2, cv2.NORMAL_CLONE)
    
    # 얼굴 합성
    output_image = np.hstack((final_output1, final_output2))
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_image)
    output_image = ImageTk.PhotoImage(output_image)
    
    if panelC is None:
        panelC = Label(root)
        panelC.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
    
    panelC.configure(image=output_image)
    panelC.image = output_image


def save_image():
    if panelC is None or not hasattr(panelC, 'image'):
        messagebox.showerror("Error", "No image to save. Please swap faces first.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    if file_path:
        image = ImageTk.getimage(panelC.image)
        image.save(file_path)
        messagebox.showinfo("Success", "Image saved successfully to {}".format(file_path))


def create_gui():
    global root, progress_bar
    
    root = Tk()
    root.title("Face Swapper")

    # Create main frame
    main_frame = Frame(root)
    main_frame.pack(padx=10, pady=10)

    # Create buttons frame
    buttons_frame = Frame(main_frame)
    buttons_frame.pack(side="top", fill="x")

    btnSelect1 = Button(buttons_frame, text="Select Image 1", command=lambda: select_image(True))
    btnSelect1.pack(side="left", padx=5)

    btnSelect2 = Button(buttons_frame, text="Select Image 2", command=lambda: select_image(False))
    btnSelect2.pack(side="left", padx=5)

    btnSwap = Button(buttons_frame, text="Swap Faces", command=lambda: threading.Thread(target=swap_faces_thread).start())
    btnSwap.pack(side="left", padx=5)

    # Create save button
    btnSave = Button(main_frame, text="Save Image", command=save_image)
    btnSave.pack(side="top", fill="x", pady=5)

    # Create progress bar
    progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=200, mode="indeterminate")
    progress_bar.pack(side="top", fill="x", pady=5)

    root.mainloop()


if __name__ == '__main__':
    create_gui()