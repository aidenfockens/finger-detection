import numpy as np
import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

template_img = cv2.imread("filename.png")  # Load your template image
template_img = template_img.astype('uint8')  # Ensure 8-bit format
template_h, template_w = template_img.shape[:2]

template_h, template_w = template_img.shape[:2]


if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()



try:
    while True:
        # Step 3: Read a frame from the stream
        ret, frame = cap.read()
        print(ret)
        if not ret:
            print("Error: Couldn't read frame")
            break

        # Step 4: Process the frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 48, 80], dtype="uint8")
        upper_skin = np.array([20, 255, 255], dtype="uint8")

        # Step 3: Create a mask that only includes skin colors
        skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

        # create an image pyramid
        # Use template matching with cross correlation to iterate over each image to see if it matches the given image
        # create a bounding box around

        # pyramid = [skin_mask]
        # for i in range(3):  # Create 3 levels of pyramid for example
        #     pyramid.append(cv2.pyrDown(pyramid[-1]))

        # for level in pyramid:
        #     res = cv2.matchTemplate(level, template_img, cv2.TM_CCOEFF_NORMED)
        #     threshold = 0.8  # Set a threshold for detecting matches
        #     loc = np.where(res >= threshold)

        #     for pt in zip(*loc[::-1]):  # Switch x and y coordinates
        #         cv2.rectangle(frame, pt, (pt[0] + template_w, pt[1] + template_h), (0, 255, 0), 2)

        # cv2.imshow('Detected Areas', frame)

  


        # # Step 4: Apply the mask to highlight skin color
        # # Use cv2.bitwise_and to apply the mask on the original frame
        # skin = cv2.bitwise_and(frame, frame, mask=skin_mask)


        # Step 5: Display the frame
        cv2.imshow('Webcam - Grayscale', skin_mask)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Step 6: Release resources
    cap.release()
    cv2.destroyAllWindows()

        # create an image pyramid
        # Use template matching with cross correlation to iterate over each image to see if it matches the given image
        # create a bounding box around  

