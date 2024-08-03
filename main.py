import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Button, font


def preprocess_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - image
    _, thresh = cv2.threshold(inverted, 160, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        margin = 25  # bounding box margin
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w += 2 * margin
        h += 2 * margin

        roi = inverted[y:y + h, x:x + w]
    else:
        roi = inverted

    roi_resized = cv2.resize(roi, (28, 28))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 28, 28, 1))

    return roi_reshaped


# video devices
def find_cap_dev():
    index = 0
    available_devices = []

    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        available_devices.append(index)
        cap.release()
        index += 1

    return available_devices


class MNISTApp:
    def __init__(self, root):
        # load model
        self.loaded_model = tf.keras.models.load_model('mnist_cnn.h5')

        # capture feed
        self.cap = cv2.VideoCapture(0)  # <---- CHANGE THIS VALUE TO CHANGE THE INPUT DEVICE ( DEFAULT IS 0 )

        # gui
        self.root = root
        self.root.title("MNIST CNN")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        self.main_label = tk.Label(root, text="Number Classifier", font=font.Font(family="Times New Roman", size=20))
        self.main_label.place(x=370, y=20)

        self.sub_label = tk.Label(root,
                                  text="_________________________________________________________________________",
                                  font=font.Font(family="Times New Roman", size=11))
        self.sub_label.place(x=150, y=60)

        # video frame
        self.video_frame = tk.Label(root)
        self.video_frame.place(x=20, y=100, height=400, width=400)

        # prediction frame
        self.predict_frame = tk.Label(root, bd=2, relief="solid", text="_",
                                      font=font.Font(family="Times New Roman", size=60))
        self.predict_frame.place(x=480, y=140, height=320, width=400)

        # button
        self.capture_button = Button(root, text="Classify", font=font.Font(family="Times New Roman", size=14),
                                     command=self.predict_frame_label)
        self.capture_button.place(x=425, y=520, height=40, width=100)

        # update video frame
        self.update_video_frame()

    # video frame update
    def update_video_frame(self):
        ret, frame = self.cap.read()
        frame = frame[80:-80, 80:-80]
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.video_frame.config(image=frame)
            self.video_frame.image = frame

        self.root.after(10, self.update_video_frame)

    # capturing and image processing
    @property
    def capture_and_process(self):
        ret, frame = self.cap.read()
        frame = frame[80:-80, 80:-80]
        if ret:
            processed_img = preprocess_img(frame)
            prediction = self.loaded_model.predict(processed_img)
            predicted_class = np.argmax(prediction)
            return predicted_class

        else:
            raise ValueError("Error in capture and process")

    # prediction frame label
    def predict_frame_label(self):
        predicted_val = self.capture_and_process
        self.predict_frame.config(text=predicted_val, font=font.Font(family="Times New Roman", size=72))

    # Release resources
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    gui = tk.Tk()
    app = MNISTApp(gui)
    gui.mainloop()
