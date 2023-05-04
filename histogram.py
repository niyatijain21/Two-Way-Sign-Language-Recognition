import cv2
import pickle

img = cv2.imread('for_hist_asl.jpeg',0)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

with open("hist", "wb") as f:
    pickle.dump(hist, f)