import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
class testingDataGen():
	def __init__(self):
		pass
	def genData(self):
		files = os.listdir('/Users/arpit/Desktop/Face_Age_Pred/UTKFace')
		
		train = {'File_Name':[], 'age' : []}
		test = {'File_Name':[], 'age' : []}
		val = {'File_Name':[], 'age' : []}
		for i in range(len(files)):
			if i < len(files)//2:
				age = int(files[i].split("_")[0])
				train['File_Name'] += ['/Users/arpit/Desktop/Face_Age_Pred/UTKFace/'+files[i]]
				train['age'] += [age]	
			elif len(files)//2<=i<(len(files)*3)//4:
				age = int(files[i].split("_")[0])
				test['File_Name'] += ['/Users/arpit/Desktop/Face_Age_Pred/UTKFace/'+files[i]]
				test['age'] += [age]
			else:
				age = int(files[i].split("_")[0])
				val['File_Name'] += ['/Users/arpit/Desktop/Face_Age_Pred/UTKFace/'+files[i]]
				val['age'] += [age]

		df_train = pd.DataFrame(train)
		df_test = pd.DataFrame(test)
		df_val = pd.DataFrame(val)

		df_train.to_csv("train.csv" )
		df_test.to_csv("test.csv")
		df_val.to_csv("val.csv")

	def orgData(self):
		#for Test
		df = pd.read_csv('test.csv')
		data_ll = df.to_dict()
		x_train = []
		t_train = data_ll['age']
		for i in range(len(data_ll['File_Name'])):
			img = Image.open(data_ll['File_Name'][i])
			img = img.resize((256, 256))
			t = transforms.PILToTensor(img)
			x_train += [t]

		#for Train 
		df = pd.read_csv('train.csv')
		data_ll = df.to_dict()
		x_test = []
		t_test = data_ll['age']
		for i in range(len(data_ll['File_Name'])):
			img = Image.open(data_ll['File_Name'][i])
			img = img.resize((256, 256))
			t = transforms.PILToTensor(img)
			x_test += [t]

		#for val
		df = pd.read_csv('val.csv')
		data_ll = df.to_dict()
		x_val = []
		t_val = data_ll['age']
		for i in range(len(data_ll['File_Name'])):
			img = Image.open(data_ll['File_Name'][i])
			img = img.resize((256, 256))
			t = transforms.PILToTensor(img)
			x_val += [t]
		return (x_train, t_train, x_test, t_test, x_val, t_val)


