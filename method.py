import torch
import torch.nn as nn


class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.net = nn.Sequential(
		nn.BatchNorm1d(num_features = 3),
		nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
		nn.ReLU(inplace = True),
		nn.MaxPool2d(kernel_size = 2, stride = 2),
		
		nn.BatchNorm1d(num_features = 16),
		nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
		nn.ReLU(inplace = True),
		nn.MaxPool2d(kernel_size = 2, stride = 2),

		nn.BatchNorm1d(num_features = 32),
		nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
		nn.ReLU(inplace = True),
		nn.MaxPool2d(kernel_size = 2, stride = 2),

		nn.BatchNorm1d(num_features = 64),
		nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
		nn.ReLU(inplace = True),
		nn.MaxPool2d(kernel_size = 2, stride = 2),

		nn.BatchNorm1d(num_features = 64),
		nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
		nn.ReLU(inplace = True),
		nn.MaxPool2d(kernel_size = 2, stride = 2),

		nn.Flatten(1, 0),
		nn.Linear(4096, 256),
		nn.ReLU(inplace = True),
		nn.Dropout(0.5),
		
		nn.Linear(256, 64),
		nn.ReLU(inplace = True),
		nn.Dropout(0.5),
		
		nn.Linear(256, 1),
		nn.ReLU(inplace = True)
		)

	def forward(self, inp):
		batchSize = inp.shape[0]
		inp = self.net(inp)
		return inp

	def model_gen(x_train, t_train, x_test, t_test, bt_size):
		model = Model()
		opt=torch.optim.SGD(list(model.parameters()), lr=0.001)
		loss_fn = torch.nn.MSELoss()
		count = 1
		b =0
		turn = 0
		dictionary = {'Name':[], 'loss_prop':[],'loss_val':[]}

		while True:

		#loss from train 
			opt.zero_grad()
			inp=torch.tensor(x_train[b:b+bt_size])
			age_pred = model(inp)
			age_true = t_train[b:b+bt_size]
			loss_to_prop = loss_fn(age_pred, age_true)
			loss_to_prop.backward()
			opt.step()

		#loss from test
			inp_train = torch.Tensor(x_test)
			age_pred_test = model(inp_test)
			age_true_test = t_test
			loss_from_train = loss_fn(age_pred_test,age_true_test)
		#saving model
			name = "/Users/arpit/Desktop/Face_Age_Pred/MODELS/weight_arr" + str(count) + ".pt"
			count += 1
			torch.save(model.state_dict(), name)
			dictionary['Name'] += [name]
			dictionary['loss_prop'] += [loss_to_prop.item()]
			dictionary['loss_val'] += [loss_from_train.item()]

			if turn >= len(x_train):
				break
			else:
				turn += 1
				if b+bt_size == len(x_train):
					b =0
				else:
					b += bt_size
			return dictionary
