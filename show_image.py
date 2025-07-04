# show_image.py
import cv2

img = cv2.imread("static/output.png")
cv2.imshow("Detected Vehicles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
