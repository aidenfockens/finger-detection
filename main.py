import numpy as np
import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

# Load all templates
zero = cv2.imread("zero_fingers2.png")  
five = cv2.imread("five_fingers2.png")
# Adding new templates
one = cv2.imread("one_fingers2.png")
two = cv2.imread("two_fingers2.png")
three = cv2.imread("three_fingers2.png")
four = cv2.imread("four_fingers2.png")
# Convert templates to grayscale and ensure they are of type uint8
templates = [zero, one, two, three, four, five]
for i, template in enumerate(templates):
    if template.ndim > 2:
        templates[i] = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    templates[i] = templates[i].astype('uint8')


cv2.imwrite("test.png", templates[1])
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

prev_frame = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame")
            break

        # Convert frame to HSV and create a skin mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype="uint8")
        upper_skin = np.array([20, 255, 255], dtype="uint8")
        skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

        # Create an image pyramid from the skin mask
        pyramid = [skin_mask]  # Start with the skin mask as the first level of the pyramid
        scaling_factor = 1.2  # For example, reduce size by a factor of 1.2
        for i in range(5):  # Create more levels for the pyramid
            new_width = int(pyramid[-1].shape[1] / scaling_factor)
            new_height = int(pyramid[-1].shape[0] / scaling_factor)
            next_level = cv2.resize(pyramid[-1], (new_width, new_height))
            pyramid.append(next_level)

        best_match = {"value": -1, "loc": None, "scale": 1, "template": None}

        # Perform template matching for each template in the pyramid levels
        count = -1
        for level in pyramid:
            count += 1
            current_scale = scaling_factor ** count
            for template in templates:
                res = cv2.matchTemplate(level, template, cv2.TM_CCOEFF_NORMED)
                max_val, _, max_loc, _ = cv2.minMaxLoc(res)
                if max_val > best_match["value"]:
                    best_match.update({"value": max_val, "loc": max_loc, "scale": current_scale, "template": template})
                cv2.imshow(f"level_zero{count} and {template}", level)

        # If a match is found, display it on the original frame
        print(best_match["value"])
        if best_match["value"] > 0.7:  # Adjust threshold as needed
            original_loc = (int(best_match["loc"][0] * best_match["scale"]), int(best_match["loc"][1] * best_match["scale"]))
            template = best_match["template"]
            original_size = (int(template.shape[1] * best_match["scale"]), int(template.shape[0] * best_match["scale"]))
            top_left = original_loc
            bottom_right = (top_left[0] + original_size[0], top_left[1] + original_size[1])

            # Convert the original frame to grayscale
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Draw the rectangle on the grayscale image
            cv2.rectangle(frame_grayscale, top_left, bottom_right, (255, 0, 0), 2)
            cv2.imshow("Best Match on Original", frame_grayscale)

        prev_frame = frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
