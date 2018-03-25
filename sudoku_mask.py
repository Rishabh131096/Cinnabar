import cv2
import numpy as np

#--------------------- OCR ---------------------------------
img = cv2.imread('digits.png',0)
gray=img 
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
 
# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
 
# Now we prepare train_data and test_data.
traini = x.reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,500)[:,np.newaxis]
print(traini.shape)
# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(traini,cv2.ml.ROW_SAMPLE,train_labels)
#---------------------------------------------------------------


img1 = cv2.imread("su.png",1)
img= cv2.imread("su.png",0)

# Adaptive thresholding
thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,201,2)

_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


#--------------------- Isolating the Sudoku grid
cnt = max(contours, key = lambda x: cv2.contourArea(x))  # finding largest contour
cv2.imshow("Thresh",thresh)

hull = cv2.convexHull(cnt) # taking convex hull
print(hull)
IMG=np.zeros(img.shape)

cv2.drawContours(IMG,[hull],0,(255,255,255),2)
IMG=cv2.GaussianBlur(IMG,(11,11),0)

IMG = np.float32(IMG)

corners = cv2.goodFeaturesToTrack(IMG, 4, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
	print(corner[0])
	x,y=corner.ravel()
	img1 = cv2.circle(img1,(x,y), 2, (0,0,255))

IMG = np.uint8(IMG)
cv2.imshow("Corners",img1)


#--------------------- Finding Perspective Transform
hull = cv2.convexHull(corners)
print(hull)

pts1 = np.float32(hull) # corners in the image
pts2 = np.float32([[600,600],[0,600],[0,0],[600,0]]) # ideal image

M = cv2.getPerspectiveTransform(pts1,pts2)

img = cv2.warpPerspective(img,M,(600,600))
cv2.imshow("Warped img",img)


img1 = cv2.warpPerspective(img1,M,(600,600))
cv2.imshow("Warped img colour",img1)


#--------------------- Ignoring Grid Lines in image
thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,9)

_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cnt = max(contours, key = lambda x: cv2.contourArea(x))

x,y,W,H = cv2.boundingRect(cnt)

thresh2=(255-thresh)

# Canny edge detection
edges = cv2.Canny(thresh2,50,150,apertureSize = 5)

# Mask for lines
img3 = np.zeros(img.shape)

###### Hough Line Detection
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for l in lines:
	for rho,theta in l:
		print("hi")
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		cv2.line(img3,(x1,y1),(x2,y2),(255,255,255),10)

######## Lines for edges
cv2.line(img3,(0,0),(thresh.shape[0],0),(255,255,255),10)
cv2.line(img3,(0,0),(0,thresh.shape[1]),(255,255,255),10)
cv2.line(img3,(thresh.shape[0]-3,thresh.shape[1]-3),(thresh.shape[0]-3,0),(255,255,255),10)
cv2.line(img3,(thresh.shape[0]-3,thresh.shape[1]-3),(0,thresh.shape[1]-3),(255,255,255),10)		
cv2.imshow('Hough Mask',img3)

####### Removing lines with help of the Mask
thresh1=thresh
for i in range(0,thresh.shape[0]):
	for j in range(0,thresh.shape[1]):
		if(thresh[i][j]==255 and img3[i][j]==255):
		   thresh1[i][j] = 0

cv2.imshow("Pure numbers",thresh1)


######### Final 2D array
final = np.zeros((9,9))
_,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

lis=[]
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	padding=(h-w)/2
	if(padding<0):
		padding=0
		
	test=thresh[y-3:y+h+3,int(x-padding):int(x+w+padding)]
	test=cv2.resize(test,(20,20), interpolation = cv2.INTER_CUBIC)
	temp = test.reshape(-1,400).astype(np.float32)
	ret,result,neighbours,dist = knn.findNearest(temp,k=5)
	final[int(y*9/H)][int(x*9/W)] = result[0][0]
	lis.append([result,x,y,w,h])

twod=[[lis[0]]];

for x in range(1,len(lis)):
	done=0
	for q in range(0,len(twod)):
		print(q)
		if(abs(twod[q][0][2]-lis[x][2])<twod[q][0][4]):
			twod[q].append(lis[x])
			done=1
			break
	if(done==0):
		twod.append([lis[x]])
			
################ Sorting each individual row
for a in range(0,len(twod)):
	twod[a]=sorted(twod[a],key = lambda l:l[1])

############### Sorting rows
twod=sorted(twod,key = lambda s:s[0][2])

for i in twod:
	print(" ")
	for j in i:
		print(j[0],end=' ')
for i in range(0,9):
	print("")
	for j in range(0,9):
		if(final[i][j] == 0):
			print(" ",end=' ')
		else:
			print(int(final[i][j]),end=' ')
cv2.waitKey(0)