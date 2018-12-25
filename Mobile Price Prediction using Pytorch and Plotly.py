
# coding: utf-8

# ## Importing the required packages

# In[1]:


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import os
import warnings
import torch.tensor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical


# ## Parameters and Hyper Parameters

# In[2]:


inp_size = 20
hidden_layer1 = 35
hidden_layer2 = 35
num_classes = 4
num_epochs = 256
batch_size = 128
learning_rate = 0.001


# ## Loading the Datatset using pandas

# In[3]:


dataset = pd.read_csv('Mobile Price Prediction.csv')


# ## Applying feature Scaling to input columns

# In[4]:


scaler=StandardScaler()
dataset=dataset.astype(float)
X=scaler.fit_transform(dataset.drop(['price_range'],axis=1))
Y=dataset['price_range'].values
warnings.simplefilter("ignore")


# ## Creating train and test Split

# In[5]:


input_data,test_input,output_data,test_output=train_test_split(X,Y,test_size=0.25,random_state=42)


# ## Creating tensor of train and test

# In[6]:


train_dataset = Data.TensorDataset(torch.from_numpy(input_data).float(),torch.from_numpy(output_data).long())
test_dataset = Data.TensorDataset(torch.from_numpy(test_input).float(),torch.from_numpy(test_output).long())


# ## Creating DataLoader for train and test set

# In[7]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# ## Making a dictionary defining training and validation sets

# In[8]:


dataloaders = dict()
dataloaders['train'] = train_loader
dataloaders['test'] = test_loader
dataset_sizes = {'train': len(input_data), 'test': len(test_input)}


# ## Check if gpu is available or not

# In[9]:


use_gpu = torch.cuda.is_available()


# ## Defining layers of Neural Network

# In[10]:


class Net(nn.Module):
    def __init__(self, inp_size, hidden_layer1, hidden_layer2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(inp_size, hidden_layer1),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_layer2, num_classes))

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


# ## A function that calculate Accuracy and loss for each epoch

# In[11]:


def train_model(model, criterion, optimizer, num_epochs):
    f = open("Iterations.txt", "w+")
    best_model_wts = model.state_dict()
    best_val_acc = 0.0
    best_train_acc = 0.0
    train_accuracy =[]
    test_accuracy = []
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, label = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(label.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(label)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == label)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            if phase == 'train':
                train_accuracy.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                test_accuracy.append(epoch_acc)
                test_loss.append(epoch_loss)
            #Print it out Loss and Accuracy and in the file torchvision
            print('{} Loss: {:.8f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            f.write('{} Loss: {:.8f} Accuracy: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'test' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = model.state_dict()
            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
                best_model_wts = model.state_dict()
    f.close()
    print('Best val Acc: {:4f}'.format(best_val_acc))
    model.load_state_dict(best_model_wts)
    return model, best_train_acc, best_val_acc,train_accuracy,test_accuracy,train_loss,test_loss


# ## Defining object of the Net Class

# In[12]:


net = Net(inp_size, hidden_layer1, hidden_layer2, num_classes)


# ## Defining the loss and Optimizer

# In[13]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# ## Training the model

# In[14]:


if use_gpu:
    model_ft, train_acc, test_acc,train_accuracy,test_accuracy,train_loss,test_loss = train_model(net.cuda(), criterion, optimizer, num_epochs)
else:
    model_ft, train_acc, test_acc,train_accuracy,test_accuracy,train_loss,test_loss = train_model(net, criterion, optimizer, num_epochs)


# ## Saving the model

# In[15]:


torch.save(model_ft.state_dict(), 'save.pkl')


# ## Setting the credential and config for plotly

# In[16]:


plotly.tools.set_credentials_file(username='ali_mirza', api_key='AIl8g6LMPjsv8255sPy5')
plotly.tools.set_config_file(world_readable=True,sharing='public')


# ## Training Set Accuracy Vs Test Set Accuracy

# In[17]:


trace1 = go.Scatter(
    x=list(range(0,num_epochs)),
    y=train_accuracy,
    name = '<b>Train Accuracy<b>',
    connectgaps=True
)
trace2 = go.Scatter(
    x=list(range(0,num_epochs)),
    y= test_accuracy,
    name = '<b>Test Accuracy<b>',
    connectgaps=True
)

data = [trace1, trace2]

fig = dict(data=data)
py.iplot(fig,filename='Training Set Accuracy Vs Test Set Accuracy', auto_open=True)


# ## Training Set Loss Vs Test Set Loss

# In[18]:


trace1 = go.Scatter(
    x=list(range(0,num_epochs)),
    y=train_loss,
    name = '<b>Train Loss<b>',
    connectgaps=True
)
trace2 = go.Scatter(
    x=list(range(0,num_epochs)),
    y= test_loss,
    name = '<b>Test Loss<b>',
    connectgaps=True
)

data = [trace1, trace2]

fig = dict(data=data)
py.iplot(fig,filename= 'Training Set Loss Vs Test Set Loss', auto_open=True)


# ## Importing tha dataset for plots

# In[19]:


orignalDataset = pd.read_csv('Mobile Price Prediction.csv')


# ## Correlation between Price range and other features

# In[20]:


data = [go.Bar(
            x=orignalDataset.columns[0:len(orignalDataset.columns)-1].tolist(),
            y=orignalDataset.corr()['price_range'].drop('price_range').tolist()
    )]

py.iplot(data,filename='Correlation between Price range and other features', auto_open=True)


# ##  Battery Power vs Price Range

# In[21]:


price_range = orignalDataset['price_range'].unique().tolist()
price_range.sort(reverse=False)
minBatteryList = orignalDataset.groupby(['price_range'])['battery_power'].min()
maxBatteryList = orignalDataset.groupby(['price_range'])['battery_power'].max()
medianBatteryList =orignalDataset.groupby(['price_range'])['battery_power'].median()

orignalDataset[orignalDataset['price_range']==0]['ram'].sort_values().median()
headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

trace0 = go.Table(
  header = dict(
    values = [['<b>Price Range</b>'],
                  ['<b>Minimum Battery Power</b>'],
                  ['<b>Maximum Battery Power</b>'],
                  ['<b>Median Battery Power</b>']],
    line = dict(color = '#506784'),
    fill = dict(color = headerColor),
    align = ['center','center'],
    font = dict(color = 'white', size = 12)
  ),
  cells = dict(
    values = [
      [price_range],
      [minBatteryList],
      [maxBatteryList],
      [medianBatteryList]],
    line = dict(color = '#506784'),
    fill = dict(color = [rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]),
    align = ['center', 'center'],
    font = dict(color = '#506784', size = 11)
    ))

data = [trace0]

py.iplot(data,filename='Battery Power vs Price Range', auto_open=True)


# ## RAM Vs Price Range

# In[22]:


minramList = []
maxramList = []
medianramList = []
for i in price_range:
    minramList.append(orignalDataset[orignalDataset['price_range']==i]['ram'].min())
    maxramList.append(orignalDataset[orignalDataset['price_range']==i]['ram'].max())
    medianramList.append(orignalDataset[orignalDataset['price_range']==i]['ram'].median())
    
headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

trace0 = go.Table(
  header = dict(
    values = [['<b>Price Range</b>'],
              ['<b>Minimum RAM</b>'],
              ['<b>Maximum RAM</b>'],
              ['<b>Median RAM</b>'] 
             ],
    line = dict(color = '#506784'),
    fill = dict(color = headerColor),
    align = ['center','center'],
    font = dict(color = 'white', size = 12)
  ),
  cells = dict(
    values = [
      [price_range],
      [minramList],
      [maxramList],
      [medianramList]
    ],
    line = dict(color = '#506784'),
    fill = dict(color = [rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]),
    align = ['center', 'center'],
    font = dict(color = '#506784', size = 11)
    ))

data = [trace0]

py.iplot(data,filename='RAM Vs Price Range', auto_open=True)


# ## Pixel Height Vs Price Range

# In[23]:


minpxHeightList = []
maxpxHeightList = []
medianpxHeightList = []
for i in price_range:
    minpxHeightList.append(orignalDataset[orignalDataset['price_range']==i]['px_height'].min())
    maxpxHeightList.append(orignalDataset[orignalDataset['price_range']==i]['px_height'].max())
    medianpxHeightList.append(orignalDataset[orignalDataset['price_range']==i]['px_height'].median())
    
headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

trace0 = go.Table(
  header = dict(
    values = [['<b>Price Range</b>'],
              ['<b>Minimum Pixel Height</b>'],
              ['<b>Maximum Pixel Height</b>'],
              ['<b>Median Pixel Height</b>'] 
             ],
    line = dict(color = '#506784'),
    fill = dict(color = headerColor),
    align = ['center','center'],
    font = dict(color = 'white', size = 12)
  ),
  cells = dict(
    values = [
      [price_range],
      [minpxHeightList],
      [maxpxHeightList],
      [medianpxHeightList]
    ],
    line = dict(color = '#506784'),
    fill = dict(color = [rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]),
    align = ['center', 'center'],
    font = dict(color = '#506784', size = 11)
    ))

data = [trace0]

py.iplot(data,filename='Pixel Height Vs Price Range¶', auto_open=True)


# ## Pixel Width Vs Price Range

# In[24]:


minpxWidthList = []
maxpxWidthList = []
medianpxWidthList = []
for i in price_range:
    minpxWidthList.append(orignalDataset[orignalDataset['price_range']==i]['px_width'].min())
    maxpxWidthList.append(orignalDataset[orignalDataset['price_range']==i]['px_width'].max())
    medianpxWidthList.append(orignalDataset[orignalDataset['price_range']==i]['px_width'].median())
    
headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

trace0 = go.Table(
  header = dict(
    values = [['<b>Price Range</b>'],
              ['<b>Minimum Pixel Width</b>'],
              ['<b>Maximum Pixel Widyh</b>'],
              ['<b>Median Pixel Width</b>'] 
             ],
    line = dict(color = '#506784'),
    fill = dict(color = headerColor),
    align = ['center','center'],
    font = dict(color = 'white', size = 12)
  ),
  cells = dict(
    values = [
      [price_range],
      [minpxWidthList],
      [maxpxWidthList],
      [medianpxWidthList]
    ],
    line = dict(color = '#506784'),
    fill = dict(color = [rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]),
    align = ['center', 'center'],
    font = dict(color = '#506784', size = 11)
    ))

data = [trace0]

py.iplot(data,filename='Pixel Width Vs Price Range¶', auto_open=True)


# ## Price Range Vs Categorical Features

# In[25]:


#blue,dual_sim,four_g,three_g,touch_screen,wifi
phone_count =[]
for i in price_range:
    phone_count.append(orignalDataset[orignalDataset['price_range']==i][orignalDataset['blue']==1][orignalDataset['dual_sim']==1][orignalDataset['four_g']==1][orignalDataset['three_g']==1][orignalDataset['touch_screen']==1][orignalDataset['wifi']==1]['price_range'].count())
data = [go.Bar(
            x=price_range,
            y=phone_count
    )]

py.iplot(data, filename='Price Range Vs Categorical Features', auto_open=True)


# ## Implementing using Keras

# In[26]:


#Feature Scaling
scaler=StandardScaler()
dataset=dataset.astype(float)
X=scaler.fit_transform(orignalDataset.drop(['price_range',],axis=1))
Y=dataset['price_range'].values


# In[27]:


#One Hot Encoding the target Variable
Y=to_categorical(Y)


# In[28]:


#Dividing it into train and test set
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# In[29]:


#Building the Keras model
model=Sequential()
model.add(Dense(input_dim=inp_size,units=8,activation='relu'))
model.add(Dense(units=hidden_layer1,activation='relu'))
model.add(Dense(units=num_classes,activation='sigmoid'))


# In[30]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[31]:


history= model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=num_epochs,verbose=1)


# In[32]:


#Evaluating the model
model.evaluate(x_test,y_test)


# ## Training set Accuracy in Keras and Pytorch for different epoch

# In[33]:


trace1 = go.Scatter(
    x=list(range(0,num_epochs)),
    y=history.history['acc'],
    name = '<b>Train Accuracy for Keras<b>',
    connectgaps=True
)
trace2 = go.Scatter(
    x=list(range(0,num_epochs)),
    y= train_accuracy,
    name = '<b>Train Accuracy for Pytorch<b>',
    connectgaps=True
)

data = [trace1, trace2]

fig = dict(data=data)
py.iplot(fig,filename='Training set Accuracy in Keras and Pytorch for different epoch', auto_open=True)


# ## Test set Accuracy in Keras and Pytorch for different epoch

# In[34]:


trace1 = go.Scatter(
    x=list(range(0,num_epochs)),
    y=history.history['val_acc'],
    name = '<b>Test Accuracy for Keras<b>',
    connectgaps=True
)
trace2 = go.Scatter(
    x=list(range(0,num_epochs)),
    y= test_accuracy,
    name = '<b>Test Accuracy for Pytorch<b>',
    connectgaps=True
)

data = [trace1, trace2]

fig = dict(data=data)
py.iplot(fig,filename='Test set Accuracy in Keras and Pytorch for different epoch', auto_open=True)


# ## Training set Loss in Keras and Pytorch for different epoch

# In[35]:


trace1 = go.Scatter(
    x=list(range(0,num_epochs)),
    y=history.history['loss'],
    name = '<b>Training Loss for Keras<b>',
    connectgaps=True
)
trace2 = go.Scatter(
    x=list(range(0,num_epochs)),
    y= train_loss,
    name = '<b>Training Loss for Pytorch<b>',
    connectgaps=True
)

data = [trace1, trace2]

fig = dict(data=data)
py.iplot(fig,filename='Training set Loss in Keras and Pytorch for different epoch', auto_open=True)


# ## Test set Loss in Keras and Pytorch for different epoch

# In[36]:


trace1 = go.Scatter(
    x=list(range(0,num_epochs)),
    y=history.history['val_loss'],
    name = '<b>Test Loss for Keras<b>',
    connectgaps=True
)
trace2 = go.Scatter(
    x=list(range(0,num_epochs)),
    y= test_loss,
    name = '<b>Test Loss for Pytorch<b>',
    connectgaps=True
)

data = [trace1, trace2]

fig = dict(data=data)
py.iplot(fig,filename='Test set Loss in Keras and Pytorch for different epoch', auto_open=True)

