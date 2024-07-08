# SIFT Feature Matching

This project uses SIFT (Scale-Invariant Feature Transform) to detect and match features between two images. The script reads pairs of images, detects keypoints, computes descriptors, matches them using BFMatcher, and then draws the matches. The results are saved as images.


## How to Use

1. **Prepare your images:** Place your image pairs in the working directory. The images should be named as Single_"name".png and Full_"name".png.
2. **Install dependencies:** Run the following command to install the required libraries:
```sh
pip install -r requirements.txt
```
3. **Run the script:** Execute the Python script to process the images.
```sh
python Feature_matching.py
```
4. **Check the results:** The output images with detected matches will be saved in the Result directory.