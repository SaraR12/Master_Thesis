from PathPrediction.model import PathPrediction
from torch import nn, cuda, optim
import torch
from PathPrediction.generateTrainingData import dataGenerator
import numpy as np

M = 8
model = PathPrediction(M)

device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'
model.to(device)

loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

train, train_truth, val, val_truth = dataGenerator(int(1))

epochs = 10
losses = []
nCorrectPreds = 0
for epoch in range(1, epochs + 1):

    for prolog, epilog in zip(train, train_truth):
        prolog = np.array(prolog)
        prolog = np.expand_dims(prolog, axis=0)
        prediction = model(torch.FloatTensor(prolog[None, ...]))
        print(prediction)
        loss = loss_fn(prediction, epilog)
        losses.append(loss.item())

        # Compute the prediction label
        predictedLabel = prediction.argmax(dim=1)

        # Add 1 to nCorrectPreds if the predicted label is correct
        nCorrectPreds += torch.sum(predictedLabel == epilog).item()

        # Perform backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()