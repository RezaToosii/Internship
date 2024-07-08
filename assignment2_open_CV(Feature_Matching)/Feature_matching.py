import cv2
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

img1 = cv2.imread('Single.png', 0)
img2 = cv2.imread('Full.png', 0)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda val: val.distance)

out0 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=0)
out2 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

os.makedirs('./Result', exist_ok=True)
path = f'{os.getcwd()}\Result'
plt.imshow(out0)
plt.savefig(f'{path}\output1_matches.png', dpi=300)

plt.imshow(out2)
plt.savefig(f'{path}\output2_matches.png', dpi=300)
