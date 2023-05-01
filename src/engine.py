import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.Module, optimiser: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
  """
    Performs a single training step on the provided data.
    
    Args:
      model (torch.nn.Module): The neural network model.
      dataloader (torch.utils.data.DataLoader): The data loader to use for training.
      loss_function (torch.nn.Module): The loss function to optimize.
      optimiser (torch.optim.Optimizer): The optimizer to use for updating the weights.
      device (torch.device): The device on which the model and data should be loaded.
    
    Returns:
        A tuple containing the average training loss and accuracy for the current step.
  """

  # We set the model to training mode
  model.train()

  training_loss, training_accuracy = 0, 0
  for batch, (X, y) in enumerate(dataloader):

    X, y = X.to(device), y.to(device)

    prediction = model(X)

    loss = loss_function(prediction, y)
    training_loss += loss.item() # loss.item() gives the loss on the current batch.

    optimiser.zero_grad() 

    loss.backward() # computes gradients

    optimiser.step() # updates weights

    # we get a vector of probability per class and then take the highest one
    prediction_class = torch.argmax(torch.softmax(prediction, dim=1), dim=1) 
    training_accuracy += (prediction_class == y).sum().item()/len(prediction)
  
  training_loss = training_loss/len(dataloader)
  training_accuracy = training_accuracy/len(dataloader)

  return training_loss, training_accuracy


def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
  """
    Performs a single testing step on the provided data.
    
    Args:
      model (torch.nn.Module): The neural network model.
      dataloader (torch.utils.data.DataLoader): The data loader to use for testing.
      loss_function (torch.nn.Module): The loss function to optimize.
      device (torch.device): The device on which the model and data should be loaded.
    
    Returns:
      A tuple containing the average testing loss and accuracy for the current step.
  """

  model.eval()# evaluation mode for the mode (prevent it from updating weights)

  test_loss, test_accuracy = 0, 0

  # we tell it we are in inference (prediction) mode
  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):

      X, y = X.to(device), y.to(device)

      prediction_logits = model(X)# forward pass

      loss = loss_function(prediction_logits, y)
      test_loss += loss.item()

      prediction_labels = prediction_logits.argmax(dim=1)
      test_accuracy += (prediction_labels == y).sum().item()/len(prediction_labels)
  
  test_loss = test_loss/len(dataloader)
  test_accuracy = test_accuracy/len(dataloader)

  return test_loss, test_accuracy

def train(model: torch.nn.Module, 
          training_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimiser: torch.optim.Optimizer, 
          loss_function: torch.nn.Module, 
          epochs: int, 
          device: torch.device, pretrained: bool = False)-> Dict[str, List]:
  """
    Trains a given model for the specified number of epochs and returns training and test losses and accuracies in the form of a dictionary.

    Args:
        model: The model to be trained.
        training_dataloader: A dataloader containing the training data.
        test_dataloader: A dataloader containing the test data.
        optimiser: The optimiser to be used during training.
        loss_function: The loss function to be used during training.
        epochs: The number of epochs to train the model for.
        device: The device to be used for computation.

    Returns:
        A dictionary containing the training and test losses and accuracies.
  """

  training_losses = []
  training_accuracies = []
  test_losses = []
  test_accuracies = []

  for epoch in tqdm(range(epochs)):

    training_loss, training_accuracy =  None
    test_loss, test_accuracy = None

    if pretrained == False:
      training_loss, training_accuracy = train_step(model=model, dataloader=training_dataloader, loss_function=loss_function, optimiser=optimiser, device=device)

      test_loss, test_accuracy = test_step(model=model, dataloader=training_dataloader, loss_function=loss_function, device=device)
    else: 
      training_loss, training_accuracy = train_step_efficientnet(model=model, dataloader=training_dataloader, loss_function=loss_function, optimiser=optimiser, device=device)

      test_loss, test_accuracy = test_step_efficientnet(model=model, dataloader=training_dataloader, loss_function=loss_function, device=device)

    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

  return {"training_loss":training_losses, "training_accuracies":training_accuracies, "test_losses": test_losses, "test_accuracies": test_accuracies} 

def train_step_efficientnet(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.Module, optimiser: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
  """
    Performs a single training step on the provided data.
    
    Args:
      model (torch.nn.Module): The neural network model.
      dataloader (torch.utils.data.DataLoader): The data loader to use for training.
      loss_function (torch.nn.Module): The loss function to optimize.
      optimiser (torch.optim.Optimizer): The optimizer to use for updating the weights.
      device (torch.device): The device on which the model and data should be loaded.
    
    Returns:
        A tuple containing the average training loss and accuracy for the current step.
  """

  # We set the model to training mode
  model.train()

  training_loss, training_accuracy = 0, 0
  examples_number = 0
  for batch, (X, y) in enumerate(dataloader):
    # we treat each segment's snippet (only the rgb image tho) as an example
    for segment in X:
      examples_number += 1
      X, y = segment[0].to(device), y.to(device)# 0 is the rgb image index
      prediction = model(X)

      loss = loss_function(prediction, y)
      training_loss += loss.item() # loss.item() gives the loss on the current batch.

      optimiser.zero_grad() 

      loss.backward() # computes gradients

      optimiser.step() # updates weights

      # we get a vector of probability per class and then take the highest one
      prediction_class = torch.argmax(torch.softmax(prediction, dim=1), dim=1) 
      training_accuracy += (prediction_class == y).sum().item()/len(prediction)
  
  training_loss = training_loss/examples_number
  training_accuracy = training_accuracy/examples_number

  return training_loss, training_accuracy

def test_step_efficientnet(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
  """
    Performs a single testing step on the provided data.
    
    Args:
      model (torch.nn.Module): The neural network model.
      dataloader (torch.utils.data.DataLoader): The data loader to use for testing.
      loss_function (torch.nn.Module): The loss function to optimize.
      device (torch.device): The device on which the model and data should be loaded.
    
    Returns:
      A tuple containing the average testing loss and accuracy for the current step.
  """

  model.eval()# evaluation mode for the mode (prevent it from updating weights)

  test_loss, test_accuracy = 0, 0

  # we tell it we are in inference (prediction) mode
  with torch.inference_mode():
    examples_number = 0
    for batch, (X, y) in enumerate(dataloader):
      # we treat each segment's snippet (only the rgb image tho) as an example
      for segment in X:
        examples_number += 1
        X, y = segment[0].to(device), y.to(device)
        prediction_logits = model(X)# forward pass

        loss = loss_function(prediction_logits, y)
        test_loss += loss.item()

        prediction_labels = prediction_logits.argmax(dim=1)
        test_accuracy += (prediction_labels == y).sum().item()/len(prediction_labels)
  
  test_loss = test_loss/examples_number
  test_accuracy = test_accuracy/examples_number

  return test_loss, test_accuracy
