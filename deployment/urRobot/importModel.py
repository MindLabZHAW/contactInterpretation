# To integrate this code with your specific models, you need to create a custom function for importing your models.
import numpy as np

import torch
import torch.nn as nn
import sys



class Sequence(nn.Module):
    def __init__(self, num_class = 5, network_type='main',num_features_lstm=4):
        super(Sequence, self).__init__()
        hidden_size = 50
        self.lstm = nn.LSTM(input_size = num_features_lstm*28, hidden_size= hidden_size, num_layers= 1, batch_first = True)
        self.network_type = network_type
        if self.network_type == 'main':
            self.linear = nn.Linear(hidden_size, num_class)    
        else:
            self.linear = nn.Linear(hidden_size*7, num_class)

        #self.linear2 = nn.Linear(50, num_class)

    def forward(self, input, future = 0):
        x, _ = self.lstm(input)

        if self.network_type == 'main':
            x = x[:,-1,:]
        else:
            x = torch.flatten(x, start_dim=1)

        x = self.linear(x)
        #x = self.linear2(x)
        #x = x[:,-1,:]
        #print(x.shape)
        return x

def get_output(data_ds, model):
    labels_pred = []
    model.eval()
    with torch.no_grad():
        for i in range(data_ds.data_target.shape[0]):
            x , y = data_ds.__getitem__(i)
            x = x[None, :]

            x = model(x)
            x = x.squeeze()
            #labels_pred.append(torch.Tensor.cpu(x.detach()).numpy())
            labels_pred.append(x.detach().numpy())
    #convert list type to array
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.argmax(axis=1)
    labels_true = np.array(data_ds.data_target[:])
    labels_true = labels_true.astype('int64')

    return torch.tensor(labels_pred), torch.tensor(labels_true)

def import_lstm_models(PATH:str):

	checkpoint = torch.load(PATH)

	model = Sequence(num_class = checkpoint["num_classes"], network_type = checkpoint["network_type"], num_features_lstm = checkpoint["num_features_lstm"])
	model.load_state_dict(checkpoint["model_state_dict"])

	if checkpoint["collision"]:
		labels_map = { # , is for saving data
					0: ',Collaborative_Contact',
					1: ',Collision,'
						}
		print('collision detection model is loaded!')
		
	elif checkpoint["localization"]:
		labels_map = { # , is for saving data
					0: ',Link 5',
					1: ',Link 6,'
						}
		print('localization model is loaded!')
	
	elif checkpoint["num_classes"] == 5:
		labels_map = { # , is for saving data
					0: ',Noncontact,',
					1: ',Intentional_Link5,',
					2: ',Intentional_Link6,',
					3: ',Collision_Link5,',
					4: ',Collision_Link6,',
						}
		print('5-classes model is loaded!')

	elif checkpoint["num_classes"] == 3: 
		labels_map = { # , is for saving data
					0: ',Noncontact,',
					1: ',Collaborative_Contact,',
					2: ',Collision,',
				} 
		print('collision detection with 3 classes model is loaded!')

	elif checkpoint["num_classes"] ==2:
		labels_map = { # , is for saving data
					0: ',Noncontact,',
					1: ',Contact,',
				} 
		print('contact detection model is loaded!')
		
	return model.eval(), labels_map, checkpoint["num_features_lstm"]
	
def import_lstm_models_old(PATH:str, num_classes:int, network_type:str, model_name:str):

	model = Sequence(num_class = num_classes, network_type = network_type)
	checkpoint = torch.load(PATH+model_name)
	model.load_state_dict(checkpoint["model_state_dict"])
	
	print('***  Models loaded  ***')
	return model.eval()