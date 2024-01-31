import cv2
import numpy as np

img = cv2.imread('20231006_110833.jpg')
img2 = img.copy()
imgc = img.copy()
imgtest = img.copy()
imgt = img.copy() # Gen image that for draw image in colour filtered

# Turn the targeted image into grayscale
gi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image with Gaussian
img = cv2.GaussianBlur(gi,(9,9),0)

# Use Canny method to do the edge detection
canny = cv2.Canny(img, 70, 200)

# Convert img to HSV image
imgh = cv2.cvtColor(imgc, cv2.COLOR_BGR2HSV) 

bound1 = np.array([20, 50, 50])
bound2 = np.array([40, 255, 255])

# Filt the image with the bound 1 and 2
filter_ = cv2.inRange(imgh, bound1, bound2)

contours1, _ = cv2.findContours(filter_ ,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)

con1 = 0

# Loob for find countour in the colour filter image
for contour1 in contours1:
    con1 += 1
    # Draw a bounding box around the detected contour
    x, y, w, h = cv2.boundingRect(contour1)
    cv2.rectangle(imgt, (x, y), (x + w, y + h), (0, 0, 255), 5)
    # Add text with the contour
    cv2.putText(imgt, str(con1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
print("the number of egg (by colour method) = ", con1)

# Find Coutours in the image after do the edge detection
contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("Number of Contours found = " + str(len(contours)))
con = 0

for contour in contours:
    con += 1
    # Draw a bounding box around the detected contour
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(imgtest, (x, y), (x + w, y + h), (0, 0, 255), 5)
    # Add text with the contour
    cv2.putText(imgtest, str(con), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



# Filter out small contours
min_contour_area = 1000  # Minimum contour area to consider as an egg
egg_count = 0

for contour in contours:
    
    
    if cv2.contourArea(contour) > min_contour_area:
        egg_count += 1
        # Draw a bounding box around the detected egg
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 5)
        # Add text with the egg count
        cv2.putText(img2, str(egg_count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
#ret, thresh1 = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
print("the number of egg", egg_count)
cv2.imwrite('filtered.jpg', filter_)
cv2.imwrite('object_alpha.jpg', canny)
cv2.imwrite('imagewithcounting.jpg', img2)
cv2.imwrite('withoutfilter.jpg', imgtest)
