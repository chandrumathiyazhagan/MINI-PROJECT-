# MINI-PROJECT
#          GESTURE Control Using Open CV Python

## AIM:

To develop a Python-based application utilizing OpenCV to recognize and interpret specific hand gestures from a live video feed. The recognized gestures will be mapped to predefined actions, enabling control of various devices or software applications through intuitive hand movements.

## ALGORITHM:

1. Import Necessary Libraries:

Import OpenCV (cv2) for image processing and video capture.
Import NumPy for numerical operations and array manipulation.


2. Initialize Video Capture:

Create a VideoCapture object to access the device's camera.


3. Define Regions of Interest (ROIs):

Define specific regions within the video frame to focus on hand gestures.
These ROIs can be defined using bounding boxes or color segmentation techniques.


4. Hand Detection and Tracking:

Use skin color detection or other techniques to identify the hand region within the ROI.
Employ tracking algorithms (e.g., Kalman filter, mean-shift) to follow the hand's movement across frames.


5. Feature Extraction:

Extract relevant features from the detected hand region:
Contour analysis: Calculate shape-based features like area, perimeter, and convex hull.
Moment invariants: Compute moment-based features that are invariant to rotation and translation.
Histogram of Oriented Gradients (HOG): Extract features that capture edge orientations.


6. Gesture Recognition:

Train a machine learning model (e.g., Support Vector Machine, Random Forest) on a dataset of labeled hand gestures.
Use the extracted features to classify the current hand gesture into predefined categories.


7. Action Mapping:

Map recognized gestures to specific actions or commands.
For example:
A fist gesture might trigger a "pause" command.
An open palm might trigger a "play" command.


8. Execute Actions:

Send commands to external devices or software applications based on the recognized gestures.
This can be achieved using various methods like keyboard input, mouse clicks, or API calls.


9. Display the Video Feed:

Continuously capture frames from the video stream.
Draw bounding boxes or overlays to visualize the detected hand region and recognized gestures.
Display the processed video feed on the screen.


10. Repeat the Process:

Continuously iterate through steps 3 to 9 to detect, recognize, and respond to new hand gestures in real-time.

## PROGRAM
```
def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol, _ = volRange

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

           left_landmark_list, right_landmark_list = get_left_right_landmarks(frame, processed, draw, mpHands)

            # Change brightness using left hand
            if left_landmark_list:
                left_distance = get_distance(frame, left_landmark_list)
                b_level = np.interp(left_distance, [50, 220], [0, 100])
                sbc.set_brightness(int(b_level))

            # Change volume using right hand
            if right_landmark_list:
                right_distance = get_distance(frame, right_landmark_list)
                vol = np.interp(right_distance, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                # Zoom control using thumb and middle finger
                right_zoom_distance = get_distance(frame, right_landmark_list, index1=4, index2=12)  # Thumb and Middle finger
                zoom_control(right_zoom_distance)  # Trigger zoom based on distance

            cv2.imshow('Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
```
## Zoom Control Function :
```
def zoom_control(distance):
    """Simulate browser zoom in/out using pyautogui."""
    # Define distance thresholds
    zoom_in_threshold = 70  # Adjust these values based on your hand size and distance
    zoom_out_threshold = 200

    if distance < zoom_in_threshold:
        # Simulate Ctrl + '+' for zooming in
        pyautogui.hotkey('ctrl', '+')
        print("Zooming In")
    elif distance > zoom_out_threshold:
        # Simulate Ctrl + '-' for zooming out
        pyautogui.hotkey('ctrl', '-')
        print("Zooming Out")
```
## Hand Landmark Detection:
```
def get_left_right_landmarks(frame, processed, draw, mpHands):
    left_landmark_list = []
    right_landmark_list = []

    if processed.multi_hand_landmarks:
        for idx, handlm in enumerate(processed.multi_hand_landmarks):
            height, width, _ = frame.shape
            if idx == 0:  # First hand (Assuming this is the left hand)
                left_landmark_list = [[id, int(lm.x * width), int(lm.y * height)] for id, lm in enumerate(handlm.landmark) if id == 4 or id == 8]
            elif idx == 1:  # Second hand (Assuming this is the right hand)
                right_landmark_list = [[id, int(lm.x * width), int(lm.y * height)] for id, lm in enumerate(handlm.landmark) if id == 4 or id == 8 or id == 12]
            
            draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    return left_landmark_list, right_landmark_list
```

## OUTPUT:

Before Moving the Finger:

![Picture1](https://github.com/user-attachments/assets/4ebc51e2-0281-45a4-9646-b1c221acc303)

After Moving the Finger:

![Picture2](https://github.com/user-attachments/assets/d956eab9-758c-437c-8839-adfb4e3270ae)

Volume Control:

![Picture3](https://github.com/user-attachments/assets/bbb6ef76-902b-4938-8846-38aff1532a50)


![Picture4](https://github.com/user-attachments/assets/81861228-68e0-44db-8d33-ca935be03483)

Zoom IN and OUT:


![Picture5](https://github.com/user-attachments/assets/33846f2d-14b4-4c25-9bb1-25114873fa64)


![Picture6](https://github.com/user-attachments/assets/630f5997-7d2f-4263-ad88-b6210654ef27)

## CONCLUSION:

The "Volume & Brightness Control Using OpenCV Python" project demonstrates an innovative use of computer vision for user interaction through hand gestures. 
By utilizing libraries like OpenCV , the system captures real-time video to detect hand movements, allowing users to adjust audio and brightness settings hands-free. 
This project showcases how gesture recognition can transform traditional device control into a more interactive experience. 
