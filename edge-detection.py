import numpy as np
 
img = cv2.imread("b5.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (11, 11), 0)
 
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
#img=laplacian
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
 

 
canny = cv2.Canny(img, 200, 250)
 
cv2.imshow("Image", img)
cv2.imshow("Sobelx", sobelx)
cv2.imshow("Sobely", sobely)
cv2.imshow("Laplacian", laplacian)
cv2.imshow("Canny", canny)
 
cv2.waitKey(0)
cv2.destroyAllWindows()