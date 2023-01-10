    # Functions to be used in the JN for the math seminar
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import time

def plot_result(X, Y, x_label, y_label, title, loglog = False):
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


class StudentTeacher(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.teacher = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
            nn.Softmax(dim=1)
        )
        self.student = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        teacher_output = self.teacher(x)
        student_output = self.student(x)
        return teacher_output, student_output

    def train(self, train_data, train_target, test_data, test_target, alpha, nb_epochs):
        criterion = nn.KLDivLoss()
        optimizer = optim.Adam(self.parameters())

        for epoch in range(nb_epochs):
            optimizer.zero_grad()
            teacher_output, student_output = self.forward(train_data)
            distillation_loss = criterion(nn.LogSoftmax(dim=1)(student_output), teacher_output) * self.alpha
            classification_loss = nn.CrossEntropyLoss()(student_output, train_target) * (1 - self.alpha)
            loss = distillation_loss + classification_loss
            loss.backward()
            optimizer.step()

            # Print the current loss and accuracy on the test set
            with torch.no_grad():
                student_output = self.forward(test_data)[1]
                test_loss = nn.CrossEntropyLoss()(student_output, test_target)
                test_acc = (student_output.argmax(dim=1) == test_target).float().mean()
                print(f'Epoch {epoch + 1}/{nb_epochs}, Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')



class teacher_student_MLP:
    def __init__(self, model_set_up, N_teachers, Ms, seeds):
        # true targer
        self.y_true_test = None
        self.y_true_train = None
        # those are lists of MLPs
        self.teachers = None
        self.students = None
        # those are dictionaries containing the results of the MLPs
        self.teachers_results = None
        self.students_results = None
        # class that sets up the MLP
        self.model_set_up = model_set_up
        # number of teachers
        self.N_teachers = N_teachers
        # number of uniform samples
        self.Ms = Ms
        # seeds
        self.seeds = seeds

    def train_teachers(self, x_train, y_train, x_test, y_test, epochs, seed=123, B=100, L=5, drop_out_p=0.3,
                       info=False, function):
        """
        Notice that the teacher NN are trained using a lot of data, therefore the resolution for the x_train variable
        will be high. Usually the teacher NN have access to more data or/and to better computational infrastructures
        than the students counterpart, therefore this is legit.
        """
        # store the info in a dictionary
        self.teachers_results = {
            "losses": [],
            "preds" : [],
            "diffs" : [],
            "actuals": [],
            "model" : [],
            "valid_results_dict" : [],
            "epochs" : [],
            "x_students" : [],
            "y_students" : []
        }
        # train all the teachers
        for _ in range(self.N_teachers):
            model = self.model_set_up(x_train, seed, B, L, drop_out_p, info)
            losses, preds, diffs, actuals, model, valid_results_dict, epochs = train(
                model=model,
                y_train=y_train,
                categorical_train=[],           # no categorical data in this regression problem
                continuous_train=x_train,
                y_test=y_test,
                categorical_valid=[],           # no categorical data in this regression problem
                continuous_valid=x_test,
                learning_rate=0.01,
                epochs=epochs,
                print_out_interval=epochs/10,
                continuous=True)
            self.teachers_results["losses"].append(losses)
            self.teachers_results["preds"].append(preds)
            self.teachers_results["diffs"].append(diffs)
            self.teachers_results["actuals"].append(actuals)
            self.teachers_results["model"].append(model)
            self.teachers_results["valid_results_dict"].append(valid_results_dict)
            self.teachers_results["epochs"].append(epochs)

            # capsule containing the x and y for the different m
            capsule_x = []
            capsule_y = []
            for m in self.Ms:
                # equally spaced points with a resolution of 1/M
                xs = np.arange(-0.5, 0.5, 1/m)
                # the output of the teachers, that would be the imput of the students
                x = model(xs)
                # the true value of the function in the points xs
                y = function(x)
                capsule_x.append(x)
                capsule_y.append(y)

            self.teachers_results["x_students"].append(capsule_x)
            self.teachers_results["y_students"].append(capsule_y)



    def train_students(self, epochs, B=100, L=5, drop_out_p=0.3, info=False):
        """
        We are assigning a student to each teacher!
        """
        # store the info in a dictionary
        self.students_results = {
            "losses": [],
            "preds": [],
            "diffs": [],
            "actuals": [],
            "model": [],
            "valid_results_dict": [],
            "epochs": [],
            "x_students": [],
            "y_students": []
        }

        for seed in self.seeds:
            for j in range(self.N_teachers):
                xs_per_student = self.teachers_results["x_students"][j]
                ys_per_student = self.teachers_results["y_students"][j]
                for i in range(len(self.Ms)):
                    # consider the different resolutions defined by 1/M
                    x_train, x_test, y_train, y_test = create_dataset(xs_per_student[i], ys_per_student[i])
                    model = self.model_set_up(x_train, seed, B, L, drop_out_p, info)
                    losses, preds, diffs, actuals, model, valid_results_dict, epochs = train(
                        model=model, y_train=y_train,
                        categorical_train=[],
                        continuous_train=x_train,
                        y_test=y_test,
                        categorical_valid=[],
                        continuous_valid=x_test,
                        learning_rate=0.01,
                        epochs=epochs,
                        print_out_interval=epochs / 10,
                        continuous=True)
                    # to differentiate the results are stored as (measure/item, MS, Seed)
                    self.students_results["losses"].append((losses, self.Ms[i], seed))
                    self.students_results["preds"].append((preds, self.Ms[i], seed))
                    self.students_results["diffs"].append((diffs, self.Ms[i], seed))
                    self.students_results["actuals"].append((actuals, self.Ms[i], seed))
                    self.students_results["model"].append((model, self.Ms[i], seed))
                    self.students_results["valid_results_dict"].append((valid_results_dict, self.Ms[i], seed))
                    self.students_results["epochs"].append((epochs, self.Ms[i], seed))


    def evaluate_students(self):
        """
        This function takes care to evaluate student's degree of learnign and to plot some useful graphs.
        """
        # losses
        unwrap_losses = {}
        aggregated_losses = {}
        for m in self.Ms:
            max_len = len(self.students_results["losses"][0])
            unwrap_losses[m].append([losses[0] for losses in self.students_results["losses"] if losses[1] == m])
            temp = [(losses[i],i) for losses in unwrap_losses[m] for i in range(max_len)]
            # TODO controlla come disassemblare i dati, in particolare controlla come fare la media degli elementi con
            # TODO lo stesso seed

            aggregated_losses[m] = [losses[0] for losses in temp if i == losses[1] for i in range(max_len)]




