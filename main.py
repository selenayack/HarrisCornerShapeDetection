import cv2
import numpy as np

img = cv2.imread('golgeli.jpeg')
grayImg= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = cv2.GaussianBlur(grayImg, (305,305), 0)
cv2.imshow("smooth",smooth)
division = cv2.divide(grayImg, smooth, scale=182)
canny=cv2.Canny(division,400,400)
cv2.imshow("division",division)
cv2.imshow("canny",canny)
#otsu threshold ile  otomatik eşik değerleri algılandı,division  resme uygulandı
ret , thresh = cv2.threshold(division,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


contours , hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('image', img)
cv2.imshow('thresh',thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()


def get_contour_areas(contours):

    all_areas= []

    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas



print ("Contour Areas before Sorting", get_contour_areas(contours))

sorted_contours= sorted(contours, key=cv2.contourArea, reverse= False)

print ("Contour Areas after Sorting", get_contour_areas(sorted_contours))

x0 = 0
for c in sorted_contours:
   
    print("\n")
    area = cv2.contourArea(c)
    print("Area: ")
    print(area)
    if(area > 1000):
      nImg = np.zeros(img.shape, dtype=np.uint8)
      masked = np.zeros(img.shape, dtype=np.uint8)
      cv2.drawContours(nImg, [c], -1, (255,0,0),1)  
      cv2.imshow("nImg",nImg)
      
      canny=cv2.Canny(nImg,255,255)
      cv2.imshow("canny",canny)

      gray = cv2.cvtColor(nImg, cv2.COLOR_BGR2GRAY)
      gray = np.float32(gray)
      dst = cv2.cornerHarris(gray,9,5,0.07)
      ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
      dst = np.uint8(dst)
      ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
      corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

      ix=0
      for i in range(1, len(corners)):
          print(corners[i])
          img[dst>0.1*dst.max()]=[0,0,255]
          ix+=1
          
      x,y,w,h = cv2.boundingRect(canny)
      
      if(abs(x0 - x) > 1):
        if(ix < 3):
          cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
          print("Circle")
          
        
        elif(ix == 3):
          cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
          print("Triangle")
          
        elif(ix == 4):
          cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
          print("Rectangle")
        
        elif(ix == 5):
          cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
          print("Pentagon")
          
        elif(ix == 6):
          cv2.putText(img, "Hexagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
          print("Hexagon")
        x0 = x
       
        
      cv2.imshow("img",img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
   
       
   


cv2.waitKey(0)
cv2.destroyAllWindows()


