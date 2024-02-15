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


if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

prev_frame = None

ret, background = cap.read()
background = cv2.GaussianBlur(background, (21, 21), 0)

try:
    while True:

        read, frame = cap.read()
        print(read)
        if not read:
            break

        frame_2 = frame.copy()


        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Wide range to try and fully capture an image
        lower_skin = np.array([0, 48, 80], dtype="uint8")
        upper_skin = np.array([20, 255, 255], dtype="uint8")

        skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

        # create an image pyramid to test our images with different sized frames
        # Found that using a lower value for the scaling factor provided smother results
        pyramid = [skin_mask]
        scaling_factor = 1.2 

        for i in range(8):  # Allows for closer hand detection
            # Calculate new pyramid size
            width_2 = int(pyramid[-1].shape[1] / scaling_factor)
            height_2 = int(pyramid[-1].shape[0] / scaling_factor)
            next_level = cv2.resize(pyramid[-1], (width_2, height_2))
            pyramid.append(next_level)


        # Store new values for each hand shape
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
        
        # threshold_zero = 0.64
        # threshold_one = 0.64
        # threshold_two = 0.58
        # threshold_three = 0.57
        # threshold_four = 0.57
        # threshold_five = 0.63

        threshold_zero = 0.5
        threshold_one = 0.5
        threshold_two = 0.5
        threshold_three = 0.5
        threshold_four = 0.5
        threshold_five = 0.5


        thresholds = [threshold_zero, threshold_one, threshold_two, threshold_three, threshold_four, threshold_five]

        for level in pyramid:
            count += 1

            # Was picking up small noise if the source image was too large
            if count < 2:
                continue

            current_scale = scaling_factor ** count 

            #TEMPLATE MATCHING: will determine how well each hand shape matches to the frame


            #template match zero
            res_zero = cv2.matchTemplate(level, zero, cv2.TM_CCOEFF_NORMED)
            _, max_val_zero, _, max_loc_zero = cv2.minMaxLoc(res_zero)
            if max_val_zero > best_match_val_zero:
                best_match_val_zero = max_val_zero
                best_match_loc_zero = max_loc_zero
                best_match_scale_zero = current_scale

            #template match five
            res_five = cv2.matchTemplate(level, five, cv2.TM_CCOEFF_NORMED)
            _, max_val_five, _, max_loc_five = cv2.minMaxLoc(res_five)
            if max_val_five > best_match_val_five:
                best_match_val_five = max_val_five
                best_match_loc_five = max_loc_five
                best_match_scale_five = current_scale

            #template match one
            res_one = cv2.matchTemplate(level, one, cv2.TM_CCOEFF_NORMED)
            _, max_val_one, _, max_loc_one = cv2.minMaxLoc(res_one)
            if max_val_one > best_match_val_one:
                best_match_val_one = max_val_one
                best_match_loc_one = max_loc_one
                best_match_scale_one = current_scale

            #template match two
            res_two = cv2.matchTemplate(level, two, cv2.TM_CCOEFF_NORMED)
            _, max_val_two, _, max_loc_two = cv2.minMaxLoc(res_two)
            if max_val_two > best_match_val_two:
                best_match_val_two = max_val_two
                best_match_loc_two = max_loc_two
                best_match_scale_two = current_scale

            #template match three
            res_three = cv2.matchTemplate(level, three, cv2.TM_CCOEFF_NORMED)
            _, max_val_three, _, max_loc_three = cv2.minMaxLoc(res_three)
            if max_val_three > best_match_val_three:
                best_match_val_three = max_val_three
                best_match_loc_three = max_loc_three
                best_match_scale_three = current_scale

            #template match four
            res_four = cv2.matchTemplate(level, four, cv2.TM_CCOEFF_NORMED)
            _, max_val_four, _, max_loc_four = cv2.minMaxLoc(res_four)
            if max_val_four > best_match_val_four:
                best_match_val_four = max_val_four
                best_match_loc_four = max_loc_four
                best_match_scale_four = current_scale

            # DRAWING RECTANGLE: If any values are above the threshhold, create a rectangle around the match
            # This is mostly for testing to see how shapes are detected at different levels

            if max_val_zero >= threshold_zero:
                
                top_left_zero = max_loc_zero
                bottom_right_zero = (top_left_zero[0] + zero.shape[1], top_left_zero[1] + zero.shape[0])
                cv2.rectangle(level, top_left_zero, bottom_right_zero, (255, 0, 0), 2)
            if max_val_five >= threshold_five:
                top_left = max_loc_five
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            if max_val_one >= threshold_one:
                top_left = max_loc_one
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            if max_val_two >= threshold_two:
                top_left = max_loc_two
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            if max_val_three >= threshold_three:
                top_left = max_loc_three
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            if max_val_four >= threshold_four:
                top_left = max_loc_four
                bottom_right = (top_left[0] + zero.shape[1], top_left[1] + zero.shape[0])
                cv2.rectangle(level, top_left, bottom_right, (255, 0, 0), 2)
            if count > 2:
                cv2.imshow(f"level_four{count}", level)

        # DETERMINE THE BEST MATCH: the best match is the one highest above its respective threshold. This is computed using the max(all)
        all = [best_match_val_zero, best_match_val_one, best_match_val_two, best_match_val_three, best_match_val_four, best_match_val_five]
        for i in range(len(all)):
            all[i] = all[i] - thresholds[i]
        print(all)
        best_match = max(all)
        print(best_match, best_match_val_zero, all.index(best_match))

        #Set THE BEST MATCH: figure out which one was the best match, store the required values and print it out
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

        # For frame differencing, not currently used
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, frame)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        prev_frame = frame.copy()


        skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

        # Configure the image stream if any match was found, determining a centroid, new bounding box, and writing the shape name
        if best_match_loc and (max_val_zero > threshold_zero or max_val_one > threshold_one or max_val_two > threshold_two or max_val_three > threshold_three or max_val_four > threshold_four or max_val_five > threshold_five):
            location = (int(best_match_loc[0] * best_match_scale), int(best_match_loc[1] * best_match_scale))
            original_size = (int(five.shape[1] * best_match_scale), int(five.shape[0] * best_match_scale))
            top_left = location
            bottom_right = (top_left[0] + original_size[0], top_left[1] + original_size[1])

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left = (30, 30)
            font_scale = 1
            font_color = (0, 0, 255)  
            line_type = 2

            cv2.putText(frame, f"Current Shape: {hand_state}", 
                        bottom_left, 
                        font, 
                        font_scale,
                        font_color,
                        line_type)


            # Isolate the area where there was a template match to further analyze the hand shape
            small_area = skin_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            y_coords, x_coords = np.where(small_area == 255)

            # more accurate bounding box
            if len(x_coords) > 0 and len(y_coords) > 0:
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                
                min_x += top_left[0]
                max_x += top_left[0]
                min_y += top_left[1]
                max_y += top_left[1]
                
                top_left_corner = (min_x, min_y)
                bottom_right_corner = (max_x, max_y)
                
                rectangle_color = (0, 255, 0)  
                thickness = 2  
                
                cv2.rectangle(frame, top_left_corner, bottom_right_corner, rectangle_color, thickness)

                # Determine the centroid
                centroid_x = int(np.mean(x_coords)) + top_left[0]  # Adjusting X coordinate according to the original image
                centroid_y = int(np.mean(y_coords)) + top_left[1] 
                centroid_color = (255, 0, 0)  # Green color for the centroid
                centroid_radius = 3
                cv2.circle(frame, (centroid_x, centroid_y), centroid_radius, centroid_color, -1)


        cv2.imshow("Best Match on Original", frame)
        cv2.imshow("Original", frame_2)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

    # [0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1, 1,1,1,1,1, 2,2,2,2,2, 2,2,2,2,2, 3,3,3,3,3, 3,3,3,3,3, 4,4,4,4,4, 4,4,4,4,4, 5,5,5,5,5, 5,5,5,5,5]
