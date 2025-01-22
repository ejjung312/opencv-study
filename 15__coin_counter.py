import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Scale, Button, HORIZONTAL


def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        messagebox.showerror("Error", "Please select an image file.")
        return

    blur_kernel_size = blur_slider.get()
    threshold_value = threshold_slider.get()
    erosion_iterations = erosion_slider.get()
    final_threshold_value = final_threshold_slider.get()
    
    # 커널 사이즈는 홀수로
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    global processed_image
    processed_image, coin_count = process_image(
        cv2.imread(file_path), blur_kernel_size, threshold_value, erosion_iterations, final_threshold_value
    )
    
    if processed_image is not None:
        cv2.imshow("Coin Counter", processed_image)
        cv2.waitKey(0)  # press a key to close the window !
        cv2.destroyAllWindows()
        messagebox.showinfo("Coin Count", "Number of coins detected: {}".format(coin_count))

def process_image(image, blur_kernel_size, threshold_value, erosion_iterations, final_threshold_value):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    
    # 임계값보다 크면 흰색, 아니면 검정색
    _, binary_image = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 침식
    kernel = np.ones((5,5), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=erosion_iterations)
    
    # 블러
    eroded_image = cv2.GaussianBlur(eroded_image, (9,9), 0)
    
    # 임계치
    _, eroded_image = cv2.threshold(eroded_image, final_threshold_value, 255, cv2.THRESH_BINARY)
    
    num_labels, labels = cv2.connectedComponents(eroded_image)
    
    # 컨투어 면적이 10보다 큰 값 반환
    coin_labels = [i for i in range(1, num_labels) if cv2.contourArea(np.array(np.where(labels == i)).T) > 10]
    
    output_image = image.copy()
    for label in coin_labels:
        coords = np.column_stack(np.where(labels == label))
        for coord in coords:
            cv2.circle(output_image, tuple(coord[::-1]), 5, (0,255,0), -1)
    
    coin_count = len(coin_labels)
    
    return output_image, coin_count


def save_results():
    if 'processed_image' not in globals():
        messagebox.showerror("Error", "No processed image to save.")
        return

    file_path = filedialog.asksaveasfile(defaultextension='.png', filetypes=[('PNG files', '*.png'),
                                                                            ('JPEG files', '*.jpg'),
                                                                            ('All files', "*.*")])
    
    if file_path:
        cv2.imwrite(file_path, process_image)
        messagebox.showinfo("Saved", "Image saved successfully!")


# Initialize main app window
app = tk.Tk()
app.title("Coin Counter")
app.geometry("300x450")

# Labels and sliders
label_blur = Label(app, text="Blur Kernel Size (odd number):")
label_blur.pack(pady=5)

blur_slider = Scale(app, from_=3, to=31, resolution=2, orient=HORIZONTAL)
blur_slider.set(5)
blur_slider.pack(pady=5)

label_threshold = Label(app, text="Threshold Value:")
label_threshold.pack(pady=5)

threshold_slider = Scale(app, from_=1, to=255, orient=HORIZONTAL)
threshold_slider.set(50)
threshold_slider.pack(pady=5)

label_erosion = Label(app, text="Erosion Iterations:")
label_erosion.pack(pady=5)

erosion_slider = Scale(app, from_=1, to=10, orient=HORIZONTAL)
erosion_slider.set(5)
erosion_slider.pack(pady=5)

label_final_threshold = Label(app, text="Final Threshold Value:")
label_final_threshold.pack(pady=5)

final_threshold_slider = Scale(app, from_=1, to=255, orient=HORIZONTAL)
final_threshold_slider.set(225)
final_threshold_slider.pack(pady=5)

# Buttons
btn_select_image = Button(app, text="Select Image", command=select_image)
btn_select_image.pack(pady=10)

btn_save_results = Button(app, text="Save Results", command=save_results)
btn_save_results.pack(pady=10)

# Run the app
app.mainloop()