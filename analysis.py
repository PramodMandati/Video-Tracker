import cv2
import numpy as np
import os
from PIL import Image
import json


path="faces_detected_unknown"
face_rec=cv2.face.LBPHFaceRecognizer_create()
first_time=True
faces=[]
ids=[]
tot_ids={}
mer_ids={}

lis=os.listdir(path)
lis=lis[0:len(lis)-1]

for i in lis:
	img=Image.open(path+'/'+i).convert('L')
	imgnp=np.array(img,'uint8')
	id1=(int(i.split('-')[-1].split('.')[0]))
	if first_time:
		first_time=False
		faces.append(imgnp)
		ids.append(id1)
		face_rec.train(faces,np.array(ids))
		face_rec.save('tr.yml')
		file=None
		with open(path+'/'+i,'rb') as f:
			file=f.read()
		with open(path+'/Final/'+i,'wb') as f:
			f.write(file)
		os.remove(path+'/'+i)
		tot_ids[id1]=str(i)
		mer_ids[id1]=str(i)
	else:
		face_rec.read('tr.yml')
		id2,conf=face_rec.predict(imgnp)
		if conf<80:
			image=tot_ids[id2]
			os.remove(path+'/'+i)
			image=image.split('--')[0]
			image2=i.split('--')[0]
			new_image=image+" "+image2+'--'+str(id2)+'.jpg'
			os.rename(path+'/Final/'+tot_ids[id2],path+'/Final/'+new_image)
			tot_ids[id2]=new_image
			mer_ids[id2]=str(mer_ids[id2].split('.')[0])+' '+str(i.split('.')[0])
		else:
			faces.append(imgnp)
			ids.append(id1)
			face_rec.train(faces,np.array(ids))
			face_rec.save('tr.yml')
			file=None
			with open(path+'/'+i,'rb') as f:
				file=f.read()
			with open(path+'/Final/'+i,'wb') as f:
				f.write(file)
			os.remove(path+'/'+i)
			tot_ids[id1]=str(i)
			mer_ids[id1]=str(i)
with open('analysis.json','w') as f:
	f.write(json.dumps(mer_ids))