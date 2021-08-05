import cv2
import os
import numpy as np


def colorDiff(c1,c2):
  return sum([abs(c1-c2) for c1, c2 in zip(c1, c2)])


def stateDetect(b,g,r):
  target = {"red": (255,0,0), "yellow": (255,255,0), "green":(0,255,0)}
  col = (r,g,b)
  diff = [[colorDiff(col,val),name] for name, val in target.items()]
  diff.sort()
  return (diff[0][1])


def detectTraffic(fname):
  
  image = cv2.imread(fname)
  G1 = cv2.inRange(image,(65,94,0),(105,132,56))
  G2 = cv2.inRange(image,(0,50,0),(0,90,0))
  G3 = cv2.inRange(image,(62,101,0),(141,176,25))
  G4 = cv2.inRange(image,(231,244,142),(252,255,172))

  Y1 = cv2.inRange(image,(9,111,190),(58,125,207))
  Y2 = cv2.inRange(image,(15,174,218),(83,255,255))
  
 
  R1 = cv2.inRange(image,(3,8,160),(7,11,170))
  R2 = cv2.inRange(image,(37,16,178),(67,77,254))

  mask = Y1+Y2+R1+R2+G1+G2+G3+G4
  out = cv2.Canny(mask,10,200)
  circles = cv2.HoughCircles(out, cv2.HOUGH_GRADIENT, 1, 100,param1=350,param2=6,minRadius=2,maxRadius=20)


  if circles is not None:
      
      circles = np.round(circles[0, :]).astype("int")

      for (x, y, r) in circles:
          
          B,G,R = np.array(image[y,x])
          col = stateDetect(B,G,R)
          cv2.putText(image,col,(x+r,y+r),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12),2)
          cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (255,0,0), 2)   
  
  cv2.imwrite(fname, image)
  
  cv2.waitKey(0)

if __name__ == "__main__":
  formats = ['.jpg','.JPG']
  for filename in os.listdir('.'):
    if filename[-4:] not in formats:
      continue
    
    detectTraffic(filename)

      

  
      
    
