import cv2
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


plt.imshow(out0)
plt.savefig('output1_matches.png')

plt.imshow(out2)
plt.savefig('output2_matches.png')