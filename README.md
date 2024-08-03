# MNIST CNN Number Classifier

This project is a simple graphical user interface (GUI) application for classifying handwritten digits using a pre-trained convolutional neural network (CNN) model. The application utilizes OpenCV for video capture, TensorFlow for model inference, and Tkinter for the GUI.

## Features

- **Real-Time Video Feed**: Displays the live feed from the selected video capture device.
- **Digit Classification**: Captures an image from the video feed, processes it, and predicts the handwritten digit using a pre-trained CNN model.
- **User-Friendly Interface**: Built with Tkinter, the GUI includes a live video frame, a classification result display, and a button to capture and classify images.

## Installation

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- Pillow

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JanithaB/cnn_mnist_cv2.git

2. **Install required packages**:
   
   ```bash
   pip install -r requirements.txt

3. **Place the pre-trained model**:
  - Ensure you have a pre-trained model named mnist_cnn.h5 in the same directory as the script.

### Usage

1.  **Run the application**:

     ```bash
     python mnist_app.py

2.  **Select the desired video input device**:
  - The default device index is 0. Change it if needed by updating the self.cap initialization in the MNISTApp class.

3.  **Capture and classify:**
  - Click the "Classify" button to capture the current frame and predict the digit.

### Code Overview

| Function/Class        | Description                                         |
|-----------------------|-----------------------------------------------------|
| `preprocess_img(image)` | Preprocesses the captured image for model input.   |
| `find_cap_dev()`      | Finds and lists available video capture devices.   |
| `MNISTApp`            | The main application class handles the GUI and model inference. |


### GUI Components

  - Video Frame: Displays the live video feed.
  - Prediction Frame: Shows the predicted digit.
  - Classify Button: Captures the current frame and displays the prediction.

### Acknowledgements

This project utilizes several powerful libraries and frameworks:

- **OpenCV**: Provides computer vision capabilities. [OpenCV Official Site](https://opencv.org/)
- **TensorFlow**: Used for deep learning models and tasks. [TensorFlow Official Site](https://www.tensorflow.org/)
- **Tkinter**: The GUI framework used for building the graphical user interface. [Tkinter Documentation](https://docs.python.org/3/library/tkinter.html)

