from method import Model
from testingDataGen import orgData
import torch


class percentAccuracy(Model):
	def __init__(self):
		pass		

	def calcPercent():
		# calculate error
		count = # fill the count
		x = orgData()
		x_test = x[2]
		t_test = x[3]
		model = torch.load("/Users/arpit/Desktop/Face_Age_Pred/MODELS/weight_arr" + str(count) + ".pt")
		res = model(x_test)
		loss_fn = torch.nn.MSELoss()
		loss = loss_fn(res, t_test)
		stdDev = loss.item() ** 0.5
		x_val = x[4]
		t_val = x[5]
		success = 0
		for i in range(len(x_val)):
			age_true = t_val[i]
			age_pred = model(x_val[i])
			if abs(age_pred - age_true) < stdDev:
				success += 1
		return success
