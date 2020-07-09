import cv2
import time
import numpy as np
import time

fontface =cv2.FONT_HERSHEY_SIMPLEX
fontscale = 2
fontcolor = (255, 255, 255)
cap=cv2.VideoCapture('test3.mp4')
face_cascade = cv2.CascadeClassifier('a.xml')
t1=time.time()
first_time=True
no=1
while cap.isOpened():
	ret,img=cap.read()
	if ret:
		string=time.asctime()
		string=string.replace(':','-')
		# string=string.replace(' ','_')
		cv2.putText(img,string,(10,25),fontface,1,(255,255,0),1)
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		faces=face_cascade.detectMultiScale(gray, 1.3, 5)
		# time.sleep(1)
		for (x,y,w,h) in faces:
			if time.time()-t1>1:
				first_time=False
				cv2.imwrite("faces_detected_unknown/"+str(string[11:19])+'--'+str(no)+".jpg",img[y:y+h,x:x+w])
				t1=time.time()
				no+=1
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
			cv2.putText(img,"unknown face",(x,y+h),fontface,2,(255,255,255),2)
		cv2.imshow('image', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break        
cap.release()
cv2.destroyAllWindows()