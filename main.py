import numpy as np
import cv2

cap = cv2.VideoCapture(0)

zero = cv2.imread("zero_fingers3.png")  
five = cv2.imread("five_fingers2.png")
one = cv2.imread("one_fingers3.png")
two = cv2.imread("two_fingers2.png") 
three = cv2.imread("three_fingers3.png")
four = cv2.imread("four_fingers3.png")

zero = cv2.cvtColor(zero, cv2.COLOR_BGR2GRAY)
five = cv2.cvtColor(five, cv2.COLOR_BGR2GRAY)
one = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
two = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY)
three = cv2.cvtColor(three, cv2.COLOR_BGR2GRAY)
four = cv2.cvtColor(four, cv2.COLOR_BGR2GRAY)

# zero = zero.astype('uint8')
# five = five.astype('uint8')
# one = one.astype('uint8')
# two = two.astype('uint8')
# three = three.astype('uint8')
# four = four.astype('uint8')


if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

prev_frame = None

ret, background = cap.read()
background = cv2.GaussianBlur(background, (21, 21), 0)


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

        gray_frame = cv2.GaussianBlur(frame, (21, 21), 0)


        frame_diff = cv2.absdiff(background, gray_frame)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        #skin_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        # create an image pyramid
        # Use template matching with cross correlation to iterate over each image to see if it matches the given image
        # create a bounding box around

        pyramid = [skin_mask]  # Start with the original image as the first level of the pyramid
        scaling_factor = 1.2  # For example, reduce size by a factor of 1.5

        for i in range(8):  # Create 2 more levels for the pyramid, adjust as needed
            # Calculate the new size
            new_width = int(pyramid[-1].shape[1] / scaling_factor)
            new_height = int(pyramid[-1].shape[0] / scaling_factor)
            # Resize the last image in the pyramid to create the next level
            next_level = cv2.resize(pyramid[-1], (new_width, new_height))
            # Add the new level to the pyramid
            pyramid.append(next_level)

        count =-1
        best_match_val_zero = -1
        best_match_val_five = -1
        best_match_val_one = -1
        best_match_val_two = -1
        best_match_val_three = -1
        best_match_val_four = -1
        best_match_loc_zero = None
        best_match_loc_five = None
        best_match_loc_one = None
        best_match_loc_two = None
        best_match_loc_three = None
        best_match_loc_four = None
        best_match_scale_zero = 1
        best_match_scale_five = 1
        best_match_scale_one = 1
        best_match_scale_two = 1
        best_match_scale_three = 1
        best_match_scale_four = 1
        
        threshold_zero = 0.65
        threshold_one = 0.65
        threshold_two = 0.58
        threshold_three = 0.57
        threshold_four = 0.57
        threshold_five = 0.63

        thresholds = [threshold_zero, threshold_one, threshold_two, threshold_three, threshold_four, threshold_five]

        for level in pyramid:
            count += 1

            if count < 2:
                continue

            current_scale = scaling_factor ** count 

            res_zero = cv2.matchTemplate(level, zero, cv2.TM_CCOEFF_NORMED)
            _, max_val_zero, _, max_loc_zero = cv2.minMaxLoc(res_zero)
            if max_val_zero > best_match_val_zero:
                best_match_val_zero = max_val_zero
                best_match_loc_zero = max_loc_zero
                best_match_scale_zero = current_scale

            # Perform template matching for "five"
            res_five = cv2.matchTemplate(level, five, cv2.TM_CCOEFF_NORMED)
            _, max_val_five, _, max_loc_five = cv2.minMaxLoc(res_five)
            if max_val_five > best_match_val_five:
                best_match_val_five = max_val_five
                best_match_loc_five = max_loc_five
                best_match_scale_five = current_scale

            res_one = cv2.matchTemplate(level, one, cv2.TM_CCOEFF_NORMED)
            _, max_val_one, _, max_loc_one = cv2.minMaxLoc(res_one)
            if max_val_one > best_match_val_one:
                best_match_val_one = max_val_one
                best_match_loc_one = max_loc_one
                best_match_scale_one = current_scale


            res_two = cv2.matchTemplate(level, two, cv2.TM_CCOEFF_NORMED)
            _, max_val_two, _, max_loc_two = cv2.minMaxLoc(res_two)
            if max_val_two > best_match_val_two:
                best_match_val_two = max_val_two
                best_match_loc_two = max_loc_two
                best_match_scale_two = current_scale

            res_three = cv2.matchTemplate(level, three, cv2.TM_CCOEFF_NORMED)
            _, max_val_three, _, max_loc_three = cv2.minMaxLoc(res_three)
            if max_val_three > best_match_val_three:
                best_match_val_three = max_val_three
                best_match_loc_three = max_loc_three
                best_match_scale_three = current_scale


            res_four = cv2.matchTemplate(level, four, cv2.TM_CCOEFF_NORMED)
            _, max_val_four, _, max_loc_four = cv2.minMaxLoc(res_four)
            if max_val_four > best_match_val_four:
                best_match_val_four = max_val_four
                best_match_loc_four = max_loc_four
                best_match_scale_four = current_scale

              # Set a threshold for detecting matches


            if max_val_zero >= threshold_zero:
                # print(f"Highest match of {max_val_zero} found at {max_loc_zero} in this pyramid level.")
                
                top_left_zero = max_loc_zero
                bottom_right_zero = (top_left_zero[0] + zero.shape[1], top_left_zero[1] + zero.shape[0])
                cv2.rectangle(level, top_left_zero, bottom_right_zero, (255, 0, 0), 2)
            # if count >2 and count < 5:
            #     cv2.imshow(f"level_zero{count}", level)
    
              # Set a threshold for detecting matches
            if max_val_five >= threshold_five:
                # print(f"Highest match of {max_val_five} found at {max_loc_five} in this pyramid level.")
                
                # If you want to draw a rectangle around the highest match
                top_left = max_loc_five
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            
            # if count >2 and count < 5:

            #     cv2.imshow(f"level_five{count}", level)

              # Set a threshold for detecting matches
            if max_val_one >= threshold_one:
                # print(f"Highest match of {max_val_one} found at {max_loc_one} in this pyramid level.")
                
                # If you want to draw a rectangle around the highest match
                top_left = max_loc_one
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            # if count >2 and count < 5:

            #     cv2.imshow(f"level_one{count}", level)

            if max_val_two >= threshold_two:
                # print(f"Highest match of {max_val_two} found at {max_loc_two} in this pyramid level.")
                
                # If you want to draw a rectangle around the highest match
                top_left = max_loc_two
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            # if count >2 and count < 5:

            #     cv2.imshow(f"level_two{count}", level)

            if max_val_three >= threshold_three:
                # print(f"Highest match of {max_val_three} found at {max_loc_three} in this pyramid level.")
                
                # If you want to draw a rectangle around the highest match
                top_left = max_loc_three
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            # if count >2 and count < 5:

            #     cv2.imshow(f"level_three{count}", level)

            if max_val_four >= threshold_four:
                # print(f"Highest match of {max_val_four} found at {max_loc_four} in this pyramid level.")
                
                # If you want to draw a rectangle around the highest match
                top_left = max_loc_four
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            if count > 2:

                cv2.imshow(f"level_four{count}", level)

        
        all = [best_match_val_zero, best_match_val_one, best_match_val_two, best_match_val_three, best_match_val_four, best_match_val_five]
        for i in range(len(all)):
            all[i] = all[i] - thresholds[i]
        print(all)
        best_match = max(all)
        print(best_match, best_match_val_zero, all.index(best_match))
        if best_match == (best_match_val_zero - thresholds[0]):
            hand_state = "Zero"
            best_match_loc = best_match_loc_zero
            best_match_scale = best_match_scale_zero
        elif best_match == (best_match_val_one - thresholds[1]):
            hand_state = "One"
            best_match_loc = best_match_loc_one
            best_match_scale = best_match_scale_one
        elif best_match == (best_match_val_two - thresholds[2]):
            hand_state = "two"
            best_match_loc = best_match_loc_two
            best_match_scale = best_match_scale_two
        elif best_match == (best_match_val_three - thresholds[3]):
            hand_state = "three"
            best_match_loc = best_match_loc_three
            best_match_scale = best_match_scale_three
        elif best_match == (best_match_val_four - thresholds[4]):
            hand_state = "four"
            best_match_loc = best_match_loc_four
            best_match_scale = best_match_scale_four
        else:
            hand_state = "Five"
            best_match_loc = best_match_loc_five
            best_match_scale = best_match_scale_five
            
        print(hand_state)

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

        skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if best_match_loc and (max_val_zero > threshold_zero or max_val_one > threshold_one or max_val_two > threshold_two or max_val_three > threshold_three or max_val_four > threshold_four or max_val_five > threshold_five):
            original_loc = (int(best_match_loc[0] * best_match_scale), int(best_match_loc[1] * best_match_scale))
            original_size = (int(five.shape[1] * best_match_scale), int(five.shape[0] * best_match_scale))
            top_left = original_loc
            bottom_right = (top_left[0] + original_size[0], top_left[1] + original_size[1])

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = (0, 255, 255)  # White color
            lineType = 2

            cv2.putText(frame_grayscale, f"Hand State: {hand_state}", 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)


            # Draw the rectangle on the grayscale image
            cv2.rectangle(frame_grayscale, top_left, bottom_right, (255, 0, 0), 2)

            roi = skin_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Step 2: Find white pixels' coordinates and calculate the centroid
            # Find all white (255) pixel coordinates in the ROI
            y_coords, x_coords = np.where(roi == 255)

            # Check if there are any white pixels to avoid division by zero
            if len(x_coords) > 0 and len(y_coords) > 0:
                centroid_x = int(np.mean(x_coords)) + top_left[0]  # Adjusting X coordinate according to the original image
                centroid_y = int(np.mean(y_coords)) + top_left[1]  # Adjusting Y coordinate according to the original image

                # Step 3: Draw the centroid on the grayscale frame
                centroid_color = (0, 255, 0)  # Green color for the centroid
                centroid_radius = 5
                cv2.circle(frame_grayscale, (centroid_x, centroid_y), centroid_radius, centroid_color, -1)

        cv2.imshow("Best Match on Original", frame_grayscale)


        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 5: Display the frame
        # cv2.imshow('Webcam - Grayscale', skin_mask)
        # cv2.imshow("Webcam, real imgs", frame_grayscale)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()