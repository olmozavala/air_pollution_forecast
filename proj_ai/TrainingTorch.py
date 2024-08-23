import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join

def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, 
                device, model_name, output_folder):
    cur_time = datetime.now()
    writer = SummaryWriter(join(output_folder,'logs',f'{model_name}_{cur_time.strftime("%Y%m%d-%H%M%S")}'))
    for epoch in range(num_epochs):
        model.train()
        # Loop over each batch from the training set (to update the model)
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f'{data.shape[0]*batch_idx}/{len(train_loader.dataset)}', end='\r')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        val_loss = 0
        # Loop over each batch from the test set (to evaluate the model)
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        # If it is the first epoch add the graph to the TensorBoard
        if epoch == 0:
            writer.add_graph(model, data)
            
        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        print(f'Epoch: {epoch+1}, Train loss: {loss:.4f}, Val loss: {val_loss:.4f}')

    print("Done!")
    writer.close()
    return model