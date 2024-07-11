import cv2
import os
import numpy as np


def change_background(original, background, output):
    # Create a directory named 'Result' if it doesn't exist
    os.makedirs('./Result', exist_ok=True)
    # Define the result and input paths
    result_path = f'{os.getcwd()}/result'
    input_path = f'{os.getcwd()}/input'
    # Open the original and background videos
    original_video = cv2.VideoCapture(f'{input_path}/{original}')
    background_video = cv2.VideoCapture(f'{input_path}/{background}')

    # Get the frame dimensions from the original video
    frame_width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)

    # Define the upper and lower bounds for the green color in HSV
    u_green = np.array([70, 150, 230])
    l_green = np.array([55, 70, 90])

    # Initialize the video writer to save the result
    result = cv2.VideoWriter(f'{result_path}/{output}.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                             original_video.get(cv2.CAP_PROP_FPS),
                             size)

    while True:

        # Read a frame from the original and background videos
        ret, frame = original_video.read()
        ret1, frame1 = background_video.read()

        # Break the loop if no frame is returned from the original video
        if not ret:
            break
        # Loop the background video if it ends
        if not ret1:
            background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = background_video.read()
        # Resize the background frame to match the original frame size
        frame1 = cv2.resize(frame1, (frame_width, frame_height))

        # Convert the original frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask to detect green regions in the original frame
        mask = cv2.inRange(hsv, l_green, u_green)

        # Replace 255 with 1 in the mask
        mask[mask == 255] = 1

        # Apply the mask to the original and background frames
        for i in range(3):
            frame[:, :, i] *= 1 - mask
            frame1[:, :, i] *= mask

        # Combine the masked original frame and the masked background frame
        frame2 = frame1 + frame

        # Display the result frame
        cv2.imshow("result", frame2)
        # Write the result frame to the output video
        result.write(frame2)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    original_video.release()
    cv2.destroyAllWindows()


# Run the change_background function if the script is executed directly
if __name__ == '__main__':
    change_background('original.mp4', 'background.mp4', 'output')
