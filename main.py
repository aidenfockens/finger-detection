import numpy as np
import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

zero = cv2.imread("zero_fingers2.png")  
five = cv2.imread("five_fingers2.png")

if zero.ndim > 2:
    zero = cv2.cvtColor(zero, cv2.COLOR_BGR2GRAY)

if five.ndim > 2:
    five = cv2.cvtColor(five, cv2.COLOR_BGR2GRAY)

# Ensure 'zero' is of type uint8 if it isn't already
zero = zero.astype('uint8')
five = five.astype('uint8')

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

prev_frame = None

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

        pyramid = [skin_mask]  # Start with the original image as the first level of the pyramid
        scaling_factor = 1.2  # For example, reduce size by a factor of 1.5

        for i in range(4):  # Create 2 more levels for the pyramid, adjust as needed
            # Calculate the new size
            new_width = int(pyramid[-1].shape[1] / scaling_factor)
            new_height = int(pyramid[-1].shape[0] / scaling_factor)
            # Resize the last image in the pyramid to create the next level
            next_level = cv2.resize(pyramid[-1], (new_width, new_height))
            # Add the new level to the pyramid
            pyramid.append(next_level)

        count = -1
        best_match_val = -1
        best_match_loc = None
        best_match_scale = 1

        for level in pyramid:
            count += 1
            res = cv2.matchTemplate(level, zero, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8  # Set a threshold for detecting matches
            min_val_zero, max_val_zero, min_loc, max_loc_zero = cv2.minMaxLoc(res)

            if max_val_zero > best_match_val:
                best_match_val = max_val_zero
                best_match_loc = max_loc_zero
                # Calculate the effective scale of this level relative to the original image
                best_match_scale = scaling_factor ** count


            # Check if the maximum match value is above the threshold
            if max_val_zero >= threshold:
                print(f"Highest match of {max_val_zero} found at {max_loc_zero} in this pyramid level.")
                
                # If you want to draw a rectangle around the highest match
                top_left_zero = max_loc_zero
                bottom_right_zero = (top_left_zero[0] + zero.shape[1], top_left_zero[1] + zero.shape[0])
                cv2.rectangle(level, top_left_zero, bottom_right_zero, (255, 0, 0), 2)
            cv2.imshow(f"level_zero{count}", level)
    
            res = cv2.matchTemplate(level, five, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8  # Set a threshold for detecting matches
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                print(f"Highest match of {max_val} found at {max_loc} in this pyramid level.")
                
                # If you want to draw a rectangle around the highest match
                top_left = max_loc
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            cv2.imshow(f"level_five{count}", level)


        # # Step 4: Apply the mask to highlight skin color
        # # Use cv2.bitwise_and to apply the mask on the original frame
        # skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

        if prev_frame is not None:
            # Compute the absolute difference between current and previous frame
            frame_diff = cv2.absdiff(prev_frame, frame)

            # Optional: Apply thresholding to highlight significant differences
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

            # Display the thresholded difference
            # cv2.imshow('Frame Difference', thresh)

        # Update the previous frame with the current frame for the next iteration
        prev_frame = frame.copy()

        if best_match_loc and max_val_zero >0.7:
            original_loc = (int(best_match_loc[0] * best_match_scale), int(best_match_loc[1] * best_match_scale))
            original_size = (int(zero.shape[1] * best_match_scale), int(zero.shape[0] * best_match_scale))
            top_left = original_loc
            bottom_right = (top_left[0] + original_size[0], top_left[1] + original_size[1])

            # Convert the original frame to grayscale
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Draw the rectangle on the grayscale image
            cv2.rectangle(frame_grayscale, top_left, bottom_right, (255, 0, 0), 2)
            cv2.imshow("Best Match on Original", frame_grayscale)




        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 5: Display the frame
        # cv2.imshow('Webcam - Grayscale', skin_mask)
        # cv2.imshow("Webcam, real imgs", frame_grayscale)

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



# have our binary image 
