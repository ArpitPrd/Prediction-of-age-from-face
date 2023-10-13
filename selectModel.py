import matplotlib.pyplot as plt
from method import Model
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import testingDataGen

class IdentifyModel():
	def __init__(self):
		pass
	def identify_model(self):
		x = testingDataGen.testingDataGen.orgData()
		dictionary = Model.model_gen(x[0], x[1], x[2], x[3], 2000)
		x_axis = []
		for i in range(len(dic['Name'])):
			x_axis += [i]
		y_axis1 = dictionary['loss_prop']
		y_axis2 = dictionary['loss_val']
		plt.plot(x_axis, y_axis1)
		plt.plot(x_axis, y_axis2)
		plt.xlabel('x - axis')
		plt.ylabel('y - axis')
		plt.legend()
		plt.show()
x = IdentifyModel()
x.identify_model()
