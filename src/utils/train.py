import os
import torch
from tqdm.auto import tqdm

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    #print(predicted)
    correct = (predicted == labels).sum().item()
    return correct

def evaluate_model(model, data_loader, loss_fn, precision):
    total_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(precision), batch_y.to(precision)
            
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)

            total_loss += loss.item()
            correct += calculate_accuracy(y_pred, batch_y)
            total += 1

    avg_loss = total_loss / total
    accuracy = correct / len(data_loader.dataset)
    return avg_loss, accuracy

def train_model(model, n_epochs, train_loader, test_loader,
                loss_fn, optimizer, 
                model_path, model_name,
                precision):
    
    train_loss_epoch = []
    train_accuracy_epoch = []
    test_loss_epoch = []
    test_accuracy_epoch = []

    for _ in tqdm(range(n_epochs), colour="red"):
        epoch_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(precision), batch_y.to(precision)
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += calculate_accuracy(y_pred, batch_y)
            total += 1

        train_loss_epoch.append(epoch_loss / total)
        train_accuracy_epoch.append(correct / len(train_loader.dataset))

        test_loss, test_accuracy = evaluate_model(model, test_loader, loss_fn, precision)
        test_loss_epoch.append(test_loss)
        test_accuracy_epoch.append(test_accuracy)

    history = {
        'train_loss': train_loss_epoch,
        'train_accuracy': train_accuracy_epoch,
        'test_loss': test_loss_epoch,
        'test_accuracy': test_accuracy_epoch,
    }

    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, model_name))
    return history
