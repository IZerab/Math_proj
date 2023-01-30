# Functions to be used in the JN for the math seminar
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import Subset
import seaborn as sns

import pandas as pd
import time


def plot_result(X, Y, x_label, y_label, title, loglog=False):
    """
    Support function to plot the results of the numerical experiments
    """
    if not loglog:
        plt.plot(X, Y)
        plt.grid()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.plot()
    else:
        plt.loglog(X, Y)
        plt.grid()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.plot()


class MLPRegressor(nn.Module):
    """
    Custom MLP regressor with various hyperparameters to set. It ihnerits from the standard nn module of pytorch.
    It can be customized by passing different sizes of the network. The sizes should be passed as a list of integers,
    where each integer specifies the number of neuron in the considered layer.
    """

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5, continuous=True):
        """
        Parameters:
            - Embedding sizes, list with 4 tuples with 2 values each
            - Number of continuous variables
            - Output size
            - a list containing the layer structure
            - the drouput probability
            - continuous is a flag that specifies if the values are continuous only (True)
        """
        super().__init__()
        # set the size of the network
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        # set the custom dropout probability
        self.emb_drop = nn.Dropout(p)
        # batch normalize the continuous variables to the standard unit scale
        self.bn_cont = nn.BatchNorm1d(n_cont)
        self.cont = continuous

        # it specifies the layer depth in our Network
        layerlist = []
        n_emb = sum((nf for ni, nf in emb_szs))  # sum of the number of features
        n_in = n_emb + n_cont

        # build explicitly the layers in the middle
        for i in layers:
            # the layer at the first input
            layerlist.append(nn.Linear(n_in, i))
            # activation function, in the paper they used the Relu
            layerlist.append(nn.ReLU(inplace=True))
            # normalize the batch
            layerlist.append(nn.BatchNorm1d(i))
            # probability of drop out
            layerlist.append(nn.Dropout(p))
            # number of the layer
            n_in = i

        # output layer
        layerlist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cont, x_cat=[]):
        embeddings = []
        # loop through the self-embed variables and append categorical values
        if not self.cont:
            for i, e in enumerate(self.embeds):
                # concatenate the embeddings
                embeddings.append(e(x_cat[:, i]))
            x = torch.cat(embeddings, 1)
            x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        # concatenate the encoded continuous and the categorical values
        if not self.cont:
            x = torch.cat([x, x_cont], 1)
            x = self.layers(x)
        else:
            x = torch.cat([x_cont], 1)
            x = self.layers(x)
        return x

    def train(model, y_train, categorical_train, continuous_train,
              y_test, categorical_valid, continuous_valid,
              learning_rate=0.001, epochs=300, print_out_interval=2, continuous=True):
        global criterion
        criterion = nn.MSELoss()  # we'll convert this to RMSE later
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        start_time = time.time()
        model.train()

        losses = []
        preds = []

        for i in range(epochs):
            i += 1  # Zero indexing trick to start the print out at epoch 1
            if not continuous:
                y_pred = model(categorical_train, continuous_train, continuous=continuous)
            else:
                y_pred = model(continuous_train)
            preds.append(y_pred)
            loss = torch.sqrt(criterion(y_pred, y_train))  # RMSE
            losses.append(loss)

            if i % print_out_interval == 1:
                print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

            # initialize the problem
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # take a step in the optimization problem
            optimizer.step()

        print('=' * 80)
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')  # print the last line
        print(f'Duration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

        # Evaluate model
        with torch.no_grad():
            if not continuous:
                y_val = model(categorical_valid, continuous_valid, continuous=continuous)
            else:
                y_val = model(continuous_valid)
            loss = torch.sqrt(criterion(y_val, y_test))
        print(f'RMSE: {loss:.8f}')

        # Create empty list to store my results
        preds = []
        diffs = []
        actuals = []

        for i in range(len(continuous_valid)):
            diff = np.abs(y_val[i].item() - y_test[i].item())
            pred = y_val[i].item()  # explain why you use item
            actual = y_test[i].item()

            diffs.append(diff)
            preds.append(pred)
            actuals.append(actual)

        valid_results_dict = {
            'predictions': preds,
            'diffs': diffs,
            'actuals': actuals
        }

        # Save model
        torch.save(model.state_dict(), f'./model_artifacts/first_func_{epochs}.pt')
        # Return components to use later
        return losses, preds, diffs, actuals, model, valid_results_dict, epochs


def create_dataset(x, y):
    # we get half the number of observations as out batch size
    batch_size = len(y) // 2

    # convert everything in pandas for ease of use
    df = pd.DataFrame()
    df["x"] = pd.DataFrame(x)
    df["y"] = pd.Series(y)

    # shuffle the data frame
    df = df.sample(frac=1).reset_index(drop=True)

    # reassign the variable
    x = pd.DataFrame(df["x"])
    y = df["y"]

    # get the torch tensors
    X = np.stack([x[col].values for col in x.columns], 1)
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)

    # Split the data
    test_size = int(batch_size * .2)
    x_train = X[:batch_size - test_size]
    x_test = X[batch_size - test_size:batch_size]

    y_train = y[:batch_size - test_size]
    y_test = y[batch_size - test_size:batch_size]

    return x_train, x_test, y_train, y_test


def set_up_MLP(X, seed, B, L, drop_out_p, info=False):
    # Use the model
    torch.manual_seed(seed)
    # set the embeddings
    emb_szs = [(0, 0)]
    layers = [B for _ in range(L)]
    model = MLPRegressor(emb_szs, X.shape[1], out_sz=1, layers=layers, p=drop_out_p)
    if info:
        print('[INFO] Model definition')
        print(model)
        print('=' * 80)
    return model


class StudentTeacher_classification(nn.Module):
    """
    In this implementation I used the nn.Sequential method in order to create the environment.
    """

    def __init__(self, hidden_size_s=10, hidden_size_t=30, input_size=1, output_size=1, depth=10, p=0.3, alpha=0.5):
        """
        This class implements a teacher-student setting. Both teacher and students are genereted iteratively.
        The weights are normalised.
        "hidden_size_*": represents the number of units in each hidden layer. s stands for student, t for teacher
        "depth": is the number of hidden layers
        We used ReLU as the activation and we perform weight normalisation finally, it creates a Sequential container
        which contains all layers and define it as a model.
        """
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = torch.tensor(alpha).to(device)

        # build explicitly the layers in the middle
        layers_student = []
        layers_teacher = []

        # teacher
        for i in range(depth - 1):
            layers_teacher.append(nn.Linear(hidden_size_t if i else input_size, hidden_size_t).to(device))
            layers_teacher.append(nn.ReLU())
            # probability of drop out
            layers_teacher.append(nn.Dropout(p))
            # layers_teacher.append(nn.utils.weight_norm(layers_teacher[-2], dim=None))

        # add the softmax
        layers_teacher.append(nn.Linear(hidden_size_t, output_size).to(device))
        layers_teacher.append(nn.Softmax(dim=1).to(device))
        self.teacher = nn.Sequential(*layers_teacher)

        # student
        for i in range(depth - 1):
            layers_student.append(nn.Linear(hidden_size_s if i else input_size, hidden_size_s).to(device))
            layers_student.append(nn.ReLU())
            # probability of drop out
            layers_teacher.append(nn.Dropout(p))
            # layers_student.append(nn.utils.weight_norm(layers_student[-2], dim=None))

        # the dimension of the softmax is 10 since we are dealing with the MNIST problem
        layers_student.append((nn.Linear(hidden_size_s, output_size)).to(device))
        layers_student.append(nn.Softmax(dim=1).to(device))
        self.student = nn.Sequential(*layers_student)

    def forward(self, x):
        teacher_output = self.teacher(x)
        student_output = self.student(x)
        return teacher_output, student_output

    def train(self, train_dataloader, test_dataloader, number_epochs):
        criterion = nn.KLDivLoss()
        optimizer = optim.Adam(self.student.parameters())
        # use the GPU!
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        number_epochs = torch.tensor(number_epochs).to(device)

        all_acc = []
        all_losses = []
        for epoch in range(number_epochs):
            # we are using the dataloaders with minibatches

            accuracies = []
            losses = []

            for idx, (train_data, train_target) in enumerate(train_dataloader):
                # pass the training tensors to the GPU
                train_data = train_data.reshape(-1, 28 * 28)
                train_data, train_target = train_data.to(device), train_target.to(device)
                # forward passage in the network
                teacher_output, student_output = self.forward(train_data)
                # compute the loss
                distillation_loss = criterion(nn.LogSoftmax(dim=1)(student_output), teacher_output) * self.alpha
                classification_loss = nn.CrossEntropyLoss()(student_output, train_target) * (1 - self.alpha)
                loss = distillation_loss + classification_loss
                # backpropagation
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for test_data, test_target in test_dataloader:
                    # passing the tensors to the cpu
                    test_data, test_target = test_data.to(device), test_target.to(device)
                    test_data = test_data.reshape(-1, 28 * 28)
                    student_output = self.forward(test_data)[1]
                    test_loss = nn.CrossEntropyLoss()(student_output, test_target)
                    losses.append(test_loss)
                    test_acc = (student_output.argmax(dim=1) == test_target).float().mean()
                    accuracies.append(test_acc)

            all_acc.append(torch.mean(torch.tensor(accuracies)))
            all_losses.append(torch.mean(torch.tensor(losses)))

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{number_epochs}, Loss: {all_losses[epoch]:.4f}, Acc: {all_acc[epoch]:.4f}')

        return all_losses, all_acc

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            student_output = self.student(x)
            prediction = student_output.argmax(dim=1)
        return prediction


def build_mnist_dataset():
    transform = transforms.ToTensor()

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # get a subset
    trainset = Subset(trainset, indices=range(len(trainset) // 10))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=600, shuffle=True)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # get a subset
    testset = Subset(testset, indices=range(len(testset) // 5))
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

    return trainloader, testloader

class ModelEvaluator:
    def __init__(self, x_train, x_test, y_train, y_test, hidden_size_s, T):
        self.StudentTeacher = StudentTeacher
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.hidden_size_s = hidden_size_s
        self.T = T
        self.models = {}

    def train_models(self):
        for B in self.hidden_size_s:
            self.models[B] = []
            accuracy = []
            for seed in range(self.T):
                model = self.StudentTeacher(hidden_size_student=B)
                model.fit(self.x_train, self.y_train)
                accuracy.append(model.evaluate(self.x_test, self.y_test))
                self.models[B].append(model)
            print("2-Norm loss for B = {}: {:.4f}".format(B, np.mean(accuracy)))

    def plot_models(self, R):
        sns.set()
        # plt.figure(figsize=(12, 8))
        x = np.linspace(-0.5, 0.5, int(R))
        for B in self.hidden_size_s:
            nested_y = []
            for model in self.models[B]:
                y = model.predict(x)
                nested_y.append(y)
            num_lists = len(nested_y)
            y_mean = [sum(col) / num_lists for col in zip(*nested_y)]
            y_err = np.std(y_mean)
            plt.errorbar(x, y_mean, yerr=y_err, fmt='o', ecolor='lightgray', markersize=5, capsize=2,
                         label="B = {}".format(B))
            plt.plot(x, y_mean, '-', linewidth=2, label="B = {}".format(B))
        plt.legend()
        plt.xlabel("Input Value")
        plt.ylabel("Prediction")
        plt.title("Model Predictions for Different Network Widths")


######################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StudentNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, output_size=1, depth=6, dropout_p=0.3):
        super(StudentNet, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        self.fc.append(nn.BatchNorm1d(hidden_size))
        self.fc.append(nn.Dropout(dropout_p))
        for i in range(depth - 2):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
            self.fc.append(nn.BatchNorm1d(hidden_size))
            self.fc.append(nn.Dropout(dropout_p))
        self.fc.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        self.to("cuda")
        x = x.to("cuda")
        for i in range(0, len(self.fc) - 2, 3):
            x = torch.relu(self.fc[i](x))
            x = self.fc[i + 1](x)
            x = self.fc[i + 2](x)
        x = self.fc[-2](x)
        x = self.fc[-1](x)
        return x

    def fit(self, X, y, epochs=10, lr=0.01, verbose=True, validation_data=None):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        running_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss

            if verbose:
                if validation_data is not None:
                    X_val, y_val = validation_data
                    with torch.no_grad():
                        val_outputs = self.forward(X_val)
                        val_loss = criterion(val_outputs, y_val)
                    print(f'Epoch {epoch + 1}/{epochs} Loss: {running_loss / len(X)} Validation Loss: {val_loss}')
                else:
                    print(f'Epoch {epoch + 1}/{epochs} Loss: {running_loss / len(X)}')

    def predict(self,x):
        return self.forward(x)


class TeacherNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, depth=6, dropout_p=0.3, second=False):
        super(TeacherNet, self).__init__()
        self.fc = nn.ModuleList()
        if second:
            # to be used then retraining the teacher
            self.fc.append(nn.Linear(input_size + 1, hidden_size))
        else:
            self.fc.append(nn.Linear(input_size, hidden_size))
        self.fc.append(nn.BatchNorm1d(hidden_size))
        self.fc.append(nn.Dropout(dropout_p))
        for i in range(depth - 2):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
            self.fc.append(nn.BatchNorm1d(hidden_size))
            self.fc.append(nn.Dropout(dropout_p))
        self.fc.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for i in range(0, len(self.fc) - 2, 3):
            x = torch.relu(self.fc[i](x))
            x = self.fc[i + 1](x)
            x = self.fc[i + 2](x)
        x = torch.relu(self.fc[-2](x))
        x = self.fc[-1](x)
        return x

    def fit(self, X, y, epochs=10, lr=0.01, verbose=True, validation_data=None):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        running_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss

            if verbose:
                if validation_data is not None:
                    X_val, y_val = validation_data
                    with torch.no_grad():
                        val_outputs = self.forward(X_val)
                        val_loss = criterion(val_outputs, y_val)
                    print(f'Epoch {epoch + 1}/{epochs} Loss: {running_loss} Validation Loss: {val_loss}')
                else:
                    print(f'Epoch {epoch + 1}/{epochs} Loss: {running_loss}')

    def predict(self,x):
        x = x.to("cuda")
        return self.forward(x).to("cuda")



class StudentTeacher:
    def __init__(self, input_size=1, hidden_size_student=100, hidden_size_teacher=200, output_size=1, depth_student=7,
                 depth_teacher=7, lr=0.001, weight_decay=1e-5, epochs=10000, device="cuda", temperature = 0.5):
        self.criterion = None
        self.temperature = temperature
        self.input_size = input_size
        self.hidden_size_student = hidden_size_student
        self.hidden_size_teacher = hidden_size_teacher
        self.output_size = output_size
        self.depth_student = depth_student
        self.depth_teacher = depth_teacher
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.alpha = 0.6
        self.loss_history = []

        # initialise the two networks
        self.teacher2 = None
        self.student = StudentNet(input_size=self.input_size, hidden_size=self.hidden_size_student,
                                  output_size=self.output_size, depth=self.depth_student).to(device)
        self.teacher = TeacherNet(input_size=self.input_size, hidden_size=self.hidden_size_teacher,
                                  output_size=self.output_size, depth=self.depth_teacher).to(device)

    def predict(self, X):
        # predict using student model
        student_preds = self.student.predict(X).to("cuda")

        # predict using teacher model
        teacher_preds = self.teacher.predict(X).to("cuda")

        # apply temperature scaling to teacher predictions
        teacher_preds = teacher_preds / self.temperature
        teacher_preds = teacher_preds.to("cuda")

        # average the predictions
        avg_preds = (student_preds + teacher_preds) / 2

        return avg_preds

    def fit(self, X, y, student_epochs=3000, teacher_epochs=3000):
        X = torch.tensor(X).to("cuda")
        y = torch.tensor(y).view(-1, 1).to("cuda")
        # fit student model on input data
        self.student.fit(X, y, epochs=student_epochs, verbose=0)

        # predict using the student model after training
        student_preds = self.student.predict(X)

        # calculate the difference between student model predictions and true target values
        student_error = student_preds - y

        # create a new dataset by adding the student error to the input data
        new_X = torch.cat((X, student_error), dim=1).to("cuda")

        self.teacher2 = TeacherNet(input_size=self.input_size, hidden_size=self.hidden_size_teacher,
                                  output_size=self.output_size, depth=self.depth_teacher, second=True).to("cuda")

        # fit teacher model on the new dataset
        self.teacher2.fit(new_X.detach(), y.detach(), epochs=teacher_epochs, verbose=0)

    def evaluate(self, X, y):
        X, y = X.cuda(), y.cuda()

        with torch.no_grad():
            # Make predictions with the student network
            y_pred = self.student(X)

            # Calculate the 2-norm between the predictions and the true labels
            error = y_pred - y

            # Calculate the mean squared error
            mse = torch.mean(error ** 2).item()

        return mse

    def plot_loss(self):
        losses = [t.cpu().detach().numpy() for t in self.loss_history]
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.show()




