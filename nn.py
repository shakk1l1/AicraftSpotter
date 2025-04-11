import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Global variables
# These variables are used to store the models and data


def load_svc_models(model):



def svc_train(data_family, label_family, data_manufacturer, label_manufacturer, model):



def n_components_selecter():
    """
    select the number of components for PCA
    :return:
    """
    n_components = float(input("number of components for pca (0 for no pca): "))
    if n_components > 1:  # if n_components is an integer it must be the number of components
        try:
            n_components = int(n_components)
        except:
            print("Invalid number of components, please enter an integer over 1 or a float under 1")
            n_components_selecter()
    # else it must be a % of the image data to keep
    return n_components


def svc_train_s(data, label, model, spca, size):
    # Convert entire datasets to tensors
    train_images = torch.empty(len(data.shape[0]), 1, size, size)
    train_labels = torch.empty(len(label.shape[0]), dtype=torch.long)
    # TODO: image.flatten() to see how to add to tensor if possible
    # Hyperparameters
    input_size = size * size

    # TODO: let the user choose the size of the NN / Hyperparameters
    hidden_size = 128
    num_layers = 2
    num_classes = 10  # 10 digits (0-9)
    learning_rate = 0.005
    num_epochs = 100

    # Create the model, loss function, and optimizer
    model = FCNNClassifier(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(train_images_flat)
        loss = criterion(outputs, train_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_accuracy = 100 * (predicted == train_labels).sum().item() / len(train_labels)

        # Save training metrics
        train_loss_list.append(loss.item())
        train_accs_list.append(train_accuracy)



def svc_predict(image_name):



def svc_test(data_family, label_family, data_manufacturer, label_manufacturer):



def svc_test_s(data, label, D_m, clf, pca):


class FCNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):

        super(FCNNClassifier, self).__init__()

        # Create a list to hold all layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        """Forward pass through the network."""
        # Forward through all but the last layer with ReLU activation
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))

        # Output layer (without activation - will be handled by loss function)
        x = self.layers[-1](x)
        return x