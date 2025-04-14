import os

import cv2
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from path import path
import numpy as np

from database import get_image_data

# use GPU if available
print("Using GPU for Neural Networks" if torch.cuda.is_available() else "Using CPU for Neural Networks, no GPU available. \nMaybe du to a non NVIDIA GPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Global variables
f_input_size, f_hidden_size, f_num_layers, f_num_classes, f_learning_rate, f_num_epochs = None, None, None, None, None, None
m_input_size, m_hidden_size, m_num_layers, m_num_classes, m_learning_rate, m_num_epochs = None, None, None, None, None, None
le_fam = None
le_man = None
# These variables are used to store the models and data


def load_nn_models(model):

    # import the functions from data_extract
    # to avoid circular imports
    from data_extract import change_size
    global f_input_size, f_hidden_size, f_num_layers, f_num_classes
    global m_input_size, m_hidden_size, m_num_layers, m_num_classes
    global le_fam, le_man
    # Check if the model files exist
    if os.path.exists(os.path.join(path, 'models/' + model, 'm_NN.pth')) and os.path.exists(os.path.join(path, 'models/' + model, 'f_NN.pth')):
        print("Model files found")
    else:
        print("Model files not found")
        print("Please train the model first")
        return 'not trained'

    print("Loading models...")
    # Load the models
    le_fam = joblib.load(os.path.join(path, 'models/' + model, 'le_fam.pkl'))
    le_man = joblib.load(os.path.join(path, 'models/' + model, 'le_man.pkl'))

    f_input_size = joblib.load(os.path.join(path, 'models/' + model, 'f_input_size.pkl'))
    f_hidden_size = joblib.load(os.path.join(path, 'models/' + model, 'f_hidden_size.pkl'))
    f_num_layers = joblib.load(os.path.join(path, 'models/' + model, 'f_num_layers.pkl'))
    f_num_classes = joblib.load(os.path.join(path, 'models/' + model, 'f_num_classes.pkl'))
    f_learning_rate = joblib.load(os.path.join(path, 'models/' + model, 'f_learning_rate.pkl'))
    f_num_epochs = joblib.load(os.path.join(path, 'models/' + model, 'f_num_epochs.pkl'))

    m_input_size = joblib.load(os.path.join(path, 'models/' + model, 'm_input_size.pkl'))
    m_hidden_size = joblib.load(os.path.join(path, 'models/' + model, 'm_hidden_size.pkl'))
    m_num_layers = joblib.load(os.path.join(path, 'models/' + model, 'm_num_layers.pkl'))
    m_num_classes = joblib.load(os.path.join(path, 'models/' + model, 'm_num_classes.pkl'))
    m_learning_rate = joblib.load(os.path.join(path, 'models/' + model, 'm_learning_rate.pkl'))
    m_num_epochs = joblib.load(os.path.join(path, 'models/' + model, 'm_num_epochs.pkl'))

    # call the function to change size and gray
    change_size(model)
    print("Models loaded")

def hyperparameters_selector(models):
    match models:
        case "cl_nn":
            print("\nselect the hyperparameters for your model:")
            print("1. hidden size (integer)")
            hidden_size = int(input("=> "))
            print("2. number of layers (integer)")
            num_layers = int(input("=> "))
            print("3. learning rate (float)")
            learning_rate = float(input("=> "))
            print("4. number of epochs (integer)")
            num_epochs = int(input("=> "))
        case _:
            print("\nunknown model")
            print("selecting base hyperparameters")
            hidden_size = 128
            num_layers = 2
            learning_rate = 0.005
            num_epochs = 100
    return hidden_size, num_layers, learning_rate, num_epochs


def nn_train(data_family, label_family, data_manufacturer, label_manufacturer, model):
    global f_input_size, f_hidden_size, f_num_layers, f_num_classes, le_fam
    global m_input_size, m_hidden_size, m_num_layers, m_num_classes, le_man
    from data_extract import size
    if input("\nDo you wish to use the same hyperparameters for both models? (y/n)").lower() == 'y':
        # Select hyperparameters for both models
        hidden_size, num_layers, learning_rate, num_epochs = hyperparameters_selector(model)
    else:
        hidden_size, num_layers, learning_rate, num_epochs = None, None, None, None
    print("\nTraining family model...")
    f_input_size, f_hidden_size, f_num_layers, f_num_classes, le_fam, f_learning_rate, f_num_epochs = nn_train_s(data_family, label_family, model, size, "f_", hidden_size, num_layers, learning_rate, num_epochs)
    print("\nTraining manufacturer model...")
    m_input_size, m_hidden_size, m_num_layers, m_num_classes, le_man, m_learning_rate, m_num_epochs = nn_train_s(data_manufacturer, label_manufacturer, model, size, "m_", hidden_size, num_layers, learning_rate, num_epochs)

    # Save the models
    joblib.dump(le_fam, os.path.join(path, 'models/' + model, 'le_fam.pkl'))
    joblib.dump(le_man, os.path.join(path, 'models/' + model, 'le_man.pkl'))

    joblib.dump(f_input_size, os.path.join(path, 'models/' + model, 'f_input_size.pkl'))
    joblib.dump(f_hidden_size, os.path.join(path, 'models/' + model, 'f_hidden_size.pkl'))
    joblib.dump(f_num_layers, os.path.join(path, 'models/' + model, 'f_num_layers.pkl'))
    joblib.dump(f_num_classes, os.path.join(path, 'models/' + model, 'f_num_classes.pkl'))
    joblib.dump(f_learning_rate, os.path.join(path, 'models/' + model, 'f_learning_rate.pkl'))
    joblib.dump(f_num_epochs, os.path.join(path, 'models/' + model, 'f_num_epochs.pkl'))

    joblib.dump(m_input_size, os.path.join(path, 'models/' + model, 'm_input_size.pkl'))
    joblib.dump(m_hidden_size, os.path.join(path, 'models/' + model, 'm_hidden_size.pkl'))
    joblib.dump(m_num_layers, os.path.join(path, 'models/' + model, 'm_num_layers.pkl'))
    joblib.dump(m_num_classes, os.path.join(path, 'models/' + model, 'm_num_classes.pkl'))
    joblib.dump(m_learning_rate, os.path.join(path, 'models/' + model, 'm_learning_rate.pkl'))
    joblib.dump(m_num_epochs, os.path.join(path, 'models/' + model, 'm_num_epochs.pkl'))

    # import last version of size and gray
    from data_extract import size, gray
    joblib.dump(size, os.path.join(path, 'models/' + model, 'size.pkl'))
    joblib.dump(gray, os.path.join(path, 'models/' + model, 'gray.pkl'))

def nn_train_s(data, label, sel_model, size, using_set, hidden_size=None, num_layers=None, learning_rate=None, num_epochs=None):

    # Convert entire datasets to tensors
    print("Converting data to tensors...")
    data = np.array(data)
    label = np.array(label)
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)

    train_images = torch.from_numpy(data)
    train_labels = torch.from_numpy(encoded_labels)

    # Hyperparameters
    print("defining hyperparameters")
    input_size = size * size
    num_classes = len(le.classes_)  # Number of unique labels

    if hidden_size is None or num_layers is None or learning_rate is None or num_epochs is None:
        # Select hyperparameters
        hidden_size, num_layers, learning_rate, num_epochs = hyperparameters_selector(sel_model)

    # Create the model, loss function, and optimizer
    model = FCNNClassifier_linear(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop - using the entire dataset at once instead of batches
    train_loss_list = []
    train_accs_list = []

    start_time = time.time()

    print(" ")
    print("Training the model...")

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(train_images.float())
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
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%')

    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds\n')

    if input("Do you want to visualize the training loss and accuracy? (y/n): ").lower() == 'y':
        # Plotting the training loss and accuracy

        plt.figure(figsize=(12, 5))

        # Plotting training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_list, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # Plotting training accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_accs_list, label='Train Accuracy', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy')
        plt.legend()

        plt.show()
    # Save the model
    if not os.path.exists(os.path.join(path, 'models/' + sel_model)):
        os.makedirs(os.path.join(path, 'models/' + sel_model))
    torch.save(model.state_dict(), os.path.join(path, 'models/' + sel_model, using_set + 'NN.pth'))
    return input_size, hidden_size, num_layers, num_classes, le, learning_rate, num_epochs



def nn_predict(image_name, model):
    """Display some test images along with their predicted labels."""
    # start the timer
    start_predict = time.time()
    # get the size and gray scale from the model to be in accordance with the training
    from data_extract import size, gray
    start_extract = time.time()

    # extract the image
    if gray:
        img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
    if img is None:
        print(f"Error loading image: {image_name}")
        return None

    # get crop values
    values = get_image_data(image_name)
    x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
    # crop and resize the image
    croped_img = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]
    croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)
    d_img = croped_img.flatten()
    d_img_array = []
    d_img_array.append(d_img)
    end_extract = time.time()

    # load the models
    NN_fam = FCNNClassifier_linear(f_input_size, f_hidden_size, f_num_layers, f_num_classes)
    NN_fam.load_state_dict(torch.load(os.path.join(path, 'models/' + model, 'f_NN.pth')))
    NN_fam.eval()  # Set the model to evaluation mode

    NN_man = FCNNClassifier_linear(m_input_size, m_hidden_size, m_num_layers, m_num_classes)
    NN_man.load_state_dict(torch.load(os.path.join(path, 'models/' + model, 'm_NN.pth')))
    NN_man.eval()  # Set the model to evaluation mode

    start_prediction = time.time()
    # Get predictions
    with torch.no_grad():
        outputs_f = NN_fam(d_img_array.float())
        _, predicted_f = torch.max(outputs_f, 1)

        outputs_m = NN_man(d_img_array.float())
        _, predicted_m = torch.max(outputs_m, 1)

    end_prediction = time.time()
    # Convert predicted labels back to original labels
    predicted_f = le_fam.inverse_transform(predicted_f.numpy())
    predicted_m = le_man.inverse_transform(predicted_m.numpy())

    # Display images and their predictions
    plt.figure(figsize=(15, 5))
    plt.imshow(croped_img, cmap='gray' if gray else None)
    plt.title(f'Predicted family: {predicted_f.item()}, Predicted manufacturer: {predicted_m.item()}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # time taken for each part
    print(f"Time taken to extract the image: {end_extract - start_extract:.2f} seconds")
    print(f"Time taken to predict the image: {end_prediction - start_prediction:.2f} seconds")
    print(f"Total time taken: {end_prediction - end_prediction:.2f} seconds")


def nn_test(data_family, label_family, data_manufacturer, label_manufacturer, model):
    print("\nTesting family model...")
    nn_test_s(data_family, label_family, "f_", model)
    print("\nTesting manufacturer model...")
    nn_test_s(data_manufacturer, label_manufacturer, "m_", model)



def nn_test_s(data, label, used_set, used_model):
    # load the models
    if used_set == "f_":
        input_size = f_input_size
        hidden_size = f_hidden_size
        num_layers = f_num_layers
        num_classes = f_num_classes
        le = le_fam
    else:
        input_size = m_input_size
        hidden_size = m_hidden_size
        num_layers = m_num_layers
        num_classes = m_num_classes
        le = le_man

    label = le.transform(label)
    model = FCNNClassifier_linear(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load(os.path.join(path, 'models/' + used_model, used_set + 'NN.pth')))
    model.eval()  # Set the model to evaluation mode

    data = np.array(data)
    label = np.array(label)

    # Convert entire datasets to tensors
    test_images_flat = torch.from_numpy(data)
    test_labels = torch.from_numpy(label)
    
    # Predict using the model
    print("Predicting test data...")
    pred_s_time = time.time()
    test_outputs = model(test_images_flat.float())
    pred_e_time = time.time()
    
    # Calculate test accuracy
    print("\nCalculating test accuracy...")
    accuracy_s_time = time.time()
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_accuracy = 100 * (test_predicted == test_labels).sum().item() / len(test_labels)
    accuracy_e_time = time.time()
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Print time taken for prediction and accuracy calculation
    print(f'\nTime taken for prediction: {pred_e_time - pred_s_time:.2f} seconds')
    print(f'Time taken for accuracy calculation: {accuracy_e_time - accuracy_s_time:.2f} seconds')
    # Print total time taken
    print(f'Total time taken: {accuracy_e_time - accuracy_e_time:.2f} seconds')
    
    
def nn_param():
    from data_extract import size, gray
    print(" ")
    print("Showing model parameters")
    print("size of the image: " + str(size))
    print("gray scale: " + str(gray))
    if f_input_size == m_input_size or f_hidden_size == m_hidden_size or f_num_layers == m_num_layers or f_num_classes == m_num_classes:
        print("The two models (family/manufacturer) have the same parameters")
        print("learning rate: " + str(f_learning_rate))
        print("hidden size: " + str(f_hidden_size))
        print("number of layers: " + str(f_num_layers))
        print("number of epochs: " + str(f_num_epochs))

    print(" ")
    print("Family model parameters:")
    print("learning rate: " + str(f_learning_rate))
    print("hidden size: " + str(f_hidden_size))
    print("number of layers: " + str(f_num_layers))
    print("number of epochs: " + str(f_num_epochs))

    print(" ")
    print("Manufacturer model parameters:")
    print("learning rate: " + str(m_learning_rate))
    print("hidden size: " + str(m_hidden_size))
    print("number of layers: " + str(m_num_layers))
    print("number of epochs: " + str(m_num_epochs))

class FCNNClassifier_linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):

        super(FCNNClassifier_linear, self).__init__()

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