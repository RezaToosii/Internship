
# Background Replacement with OpenCV

This Python script replaces the background of an original video with a new background video using OpenCV.

## Requirements

- Python 3.x
- OpenCV
- NumPy

Run the following command to install the required libraries:
```sh
pip install -r requirements.txt
```

## Usage

1. Place your original video (`original.mp4`) and background video (`background.mp4`) in the `input` directory.
2. Run the script:

```sh
python Background_Replacement.py
```

3. The output video (`output.avi`) will be saved in the `result` directory.


## Script Explanation

The `Background_Replacement.py` script performs the following steps:

1. Creates a `Result` directory to store the output video.
2. Reads the original and background videos from the `input` directory.
3. Initializes a video writer to save the result.
4. Iterates through each frame of the original video and applies a mask to detect green regions.
5. Replaces the green regions in the original video with corresponding regions from the background video.
6. Displays the result in a window and writes it to the output video.
7. Exits the loop and releases the video capture when the original video ends or if 'q' is pressed.


