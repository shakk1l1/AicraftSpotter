import os
import cv2
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightning as L
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.nn.functional import dropout
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam
from path import path
import numpy as np
from database import get_image_data
from flexible_cnn import FlexibleCNN, train_flexible_cnn

# use GPU if available
print("Using GPU for Neural Networks" if torch.cuda.is_available() else "Using CPU for Neural Networks, no GPU available. \nMaybe du to a non NVIDIA GPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Global variables
f_input_size, f_hidden_size, f_num_layers, f_num_classes, f_learning_rate, f_num_epochs, f_filtersize, f_num_filter, f_poolsize = None, None, None, None, None, None, None, None, None
m_input_size, m_hidden_size, m_num_layers, m_num_classes, m_learning_rate, m_num_epochs, m_filtersize, m_num_filter, m_poolsize = None, None, None, None, None, None, None, None, None
f_batch_size, f_dropout_prob = None, None
m_batch_size, m_dropout_prob = None, None
le_fam = None
le_man = None
f_D_m, m_D_m = None, None
# These variables are used to store the models and data


def load_nn_models(model):

    # import the functions from data_extract
    # to avoid circular imports
    from data_extract import change_size
    global f_input_size, f_hidden_size, f_num_layers, f_num_classes, f_learning_rate, f_num_epochs, f_filtersize, f_num_filter, f_poolsize
    global m_input_size, m_hidden_size, m_num_layers, m_num_classes, m_learning_rate, m_num_epochs, m_filtersize, m_num_filter, m_poolsize
    global le_fam, le_man
    global f_D_m, m_D_m
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
    f_filtersize = joblib.load(os.path.join(path, 'models/' + model, 'f_filtersize.pkl'))
    f_num_filter = joblib.load(os.path.join(path, 'models/' + model, 'f_num_filter.pkl'))
    f_poolsize = joblib.load(os.path.join(path, 'models/' + model, 'f_poolsize.pkl'))
    f_batch_size = joblib.load(os.path.join(path, 'models/' + model, 'f_batch_size.pkl'))
    f_dropout_prob = joblib.load(os.path.join(path, 'models/' + model, 'f_dropout_prob.pkl'))

    m_input_size = joblib.load(os.path.join(path, 'models/' + model, 'm_input_size.pkl'))
    m_hidden_size = joblib.load(os.path.join(path, 'models/' + model, 'm_hidden_size.pkl'))
    m_num_layers = joblib.load(os.path.join(path, 'models/' + model, 'm_num_layers.pkl'))
    m_num_classes = joblib.load(os.path.join(path, 'models/' + model, 'm_num_classes.pkl'))
    m_learning_rate = joblib.load(os.path.join(path, 'models/' + model, 'm_learning_rate.pkl'))
    m_num_epochs = joblib.load(os.path.join(path, 'models/' + model, 'm_num_epochs.pkl'))
    m_filtersize = joblib.load(os.path.join(path, 'models/' + model, 'm_filtersize.pkl'))
    m_num_filter = joblib.load(os.path.join(path, 'models/' + model, 'm_num_filter.pkl'))
    m_poolsize = joblib.load(os.path.join(path, 'models/' + model, 'm_poolsize.pkl'))
    m_batch_size = joblib.load(os.path.join(path, 'models/' + model, 'm_batch_size.pkl'))
    m_dropout_prob = joblib.load(os.path.join(path, 'models/' + model, 'm_dropout_prob.pkl'))

    f_D_m = joblib.load(os.path.join(path, 'models/' + model, 'f_D_m.pkl'))
    m_D_m = joblib.load(os.path.join(path, 'models/' + model, 'm_D_m.pkl'))

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
            filtersize = None
            num_filters = None
            poolsize = None

        case "improved_nn":
            print("\nselect the hyperparameters for your model:")
            print("1. hidden size (integer)")
            hidden_size = int(input("=> "))
            print("2. number of layers (integer)")
            num_layers = int(input("=> "))
            print("3. learning rate (float)")
            learning_rate = float(input("=> "))
            print("4. number of epochs (integer)")
            num_epochs = int(input("=> "))
            filtersize = None
            num_filters = None
            poolsize = None

        case "cc_nn":
            print("\nselect the hyperparameters for your model:")
            print("1. filtersize (integer)")
            filtersize = int(input("=> "))
            print("2. poolsize (integer)")
            poolsize = int(input("=> "))
            print("3. number of filters (integer)")
            num_filters = int(input("=> "))
            print("4. hidden size (integer)")
            hidden_size = int(input("=> "))
            print("5. number of layers (integer)")
            num_layers = int(input("=> "))
            print("6. learning rate (float)")
            learning_rate = float(input("=> "))
            print("7. number of epochs (integer)")
            num_epochs = int(input("=> "))

        case _:
            print("\nunknown model")
            print("selecting base hyperparameters")
            hidden_size = 128
            num_layers = 2
            learning_rate = 0.005
            num_epochs = 100
            filtersize = None
            num_filters = None
            poolsize = None

    # For models other than flexible_cnn, return without dropout_prob and batch_size
    return hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters, poolsize

def hyperparameters_selector_2(models):
    print("\nselect the hyperparameters for your FlexibleCNN model:")
    print("1. base filters (integer, recommended: 32-128)")
    num_filters = int(input("=> "))
    print("2. learning rate (float, recommended: 0.001-0.01)")
    learning_rate = float(input("=> "))
    print("3. number of epochs (integer, recommended: 50-200)")
    num_epochs = int(input("=> "))
    print("4. dropout probability (float, recommended: 0.2-0.5)")
    dropout_prob = float(input("=> "))
    print("5. batch size (integer, recommended: 16-128)")
    batch_size = int(input("=> "))
    # Set other parameters to None or default values
    hidden_size = num_filters  # Use num_filters as hidden_size for compatibility
    num_layers = 2  # Default value
    filtersize = 3  # Default value
    poolsize = 2  # Default value
    return hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters, poolsize, dropout_prob, batch_size

def nn_train(data_family, label_family, data_manufacturer, label_manufacturer, model):
    global f_input_size, f_hidden_size, f_num_layers, f_num_classes, le_fam, f_learning_rate, f_num_epochs, f_filtersize, f_num_filter, f_poolsize, f_batch_size, f_dropout_prob
    global m_input_size, m_hidden_size, m_num_layers, m_num_classes, le_man, m_learning_rate, m_num_epochs, m_filtersize, m_num_filter, m_poolsize, m_batch_size, m_dropout_prob
    global f_D_m, m_D_m
    from data_extract import size
    if input("\nDo you wish to use the same hyperparameters for both models? (y/n) ").lower() == 'y':
        # Select hyperparameters for both models
        if model == "flexible_cnn":
            (hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters, poolsize,
             dropout_prob, batch_size) = hyperparameters_selector_2(model)
        else:
            (hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters,
             poolsize) = hyperparameters_selector(model)
            dropout_prob = None
            batch_size = None
    else:
        (hidden_size, num_layers, learning_rate, num_epochs, filtersize,
         num_filters, poolsize, dropout_prob, batch_size) = None, None, None, None, None, None, None, None, None
    print("\nTraining family model...")
    (f_input_size, f_hidden_size, f_num_layers, f_num_classes, le_fam, f_learning_rate, f_num_epochs, f_D_m,
     f_filtersize, f_num_filter, f_poolsize) \
        = nn_train_s(data_family, label_family, model, size, "f_",
                     hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters, poolsize,
                     dropout_prob=dropout_prob, batch_size=batch_size)
    print("\nTraining manufacturer model...")
    (m_input_size, m_hidden_size, m_num_layers, m_num_classes, le_man, m_learning_rate, m_num_epochs, m_D_m,
     m_filtersize, m_num_filter, m_poolsize)\
        = nn_train_s(data_manufacturer, label_manufacturer, model, size, "m_"
                     , hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters, poolsize)

    # Save the models
    joblib.dump(le_fam, os.path.join(path, 'models/' + model, 'le_fam.pkl'))
    joblib.dump(le_man, os.path.join(path, 'models/' + model, 'le_man.pkl'))

    joblib.dump(f_input_size, os.path.join(path, 'models/' + model, 'f_input_size.pkl'))
    joblib.dump(f_hidden_size, os.path.join(path, 'models/' + model, 'f_hidden_size.pkl'))
    joblib.dump(f_num_layers, os.path.join(path, 'models/' + model, 'f_num_layers.pkl'))
    joblib.dump(f_num_classes, os.path.join(path, 'models/' + model, 'f_num_classes.pkl'))
    joblib.dump(f_learning_rate, os.path.join(path, 'models/' + model, 'f_learning_rate.pkl'))
    joblib.dump(f_num_epochs, os.path.join(path, 'models/' + model, 'f_num_epochs.pkl'))
    joblib.dump(f_filtersize, os.path.join(path, 'models/' + model, 'f_filtersize.pkl'))
    joblib.dump(f_num_filter, os.path.join(path, 'models/' + model, 'f_num_filter.pkl'))
    joblib.dump(f_poolsize, os.path.join(path, 'models/' + model, 'f_poolsize.pkl'))
    joblib.dump(f_batch_size, os.path.join(path, 'models/' + model, 'f_batch_size.pkl'))
    joblib.dump(f_dropout_prob, os.path.join(path, 'models/' + model, 'f_dropout_prob.pkl'))

    joblib.dump(m_input_size, os.path.join(path, 'models/' + model, 'm_input_size.pkl'))
    joblib.dump(m_hidden_size, os.path.join(path, 'models/' + model, 'm_hidden_size.pkl'))
    joblib.dump(m_num_layers, os.path.join(path, 'models/' + model, 'm_num_layers.pkl'))
    joblib.dump(m_num_classes, os.path.join(path, 'models/' + model, 'm_num_classes.pkl'))
    joblib.dump(m_learning_rate, os.path.join(path, 'models/' + model, 'm_learning_rate.pkl'))
    joblib.dump(m_num_epochs, os.path.join(path, 'models/' + model, 'm_num_epochs.pkl'))
    joblib.dump(m_filtersize, os.path.join(path, 'models/' + model, 'm_filtersize.pkl'))
    joblib.dump(m_num_filter, os.path.join(path, 'models/' + model, 'm_num_filter.pkl'))
    joblib.dump(m_poolsize, os.path.join(path, 'models/' + model, 'm_poolsize.pkl'))
    joblib.dump(m_batch_size, os.path.join(path, 'models/' + model, 'm_batch_size.pkl'))
    joblib.dump(m_dropout_prob, os.path.join(path, 'models/' + model, 'm_dropout_prob.pkl'))

    # Save the mean values
    joblib.dump(f_D_m, os.path.join(path, 'models/' + model, 'f_D_m.pkl'))
    joblib.dump(m_D_m, os.path.join(path, 'models/' + model, 'm_D_m.pkl'))

    # import last version of size and gray
    from data_extract import size, gray
    joblib.dump(size, os.path.join(path, 'models/' + model, 'size.pkl'))
    joblib.dump(gray, os.path.join(path, 'models/' + model, 'gray.pkl'))

def nn_train_s(data, label, sel_model, size, using_set, hidden_size=None, num_layers=None, learning_rate=None, num_epochs=None,
               filtersize=None, num_filters=None, poolsize=None, dropout_prob=None, batch_size=None):

    # Convert entire datasets to tensors
    from data_extract import gray
    print("Converting data to tensors...")
    data = np.array(data) /255

    print("calculating mean...")
    D_m = np.mean(data, axis=0)
    print("centering data...")
    data = data - D_m

    label = np.array(label)
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)

    train_images = torch.from_numpy(data)
    train_labels = torch.from_numpy(encoded_labels)

    # Hyperparameters
    print("defining hyperparameters")
    input_size = size * size
    num_classes = len(le.classes_)  # Number of unique labels

    # Default values for flexible_cnn specific parameters
    dropout_prob = 0.5
    batch_size = 32

    if hidden_size is None or num_layers is None or learning_rate is None or num_epochs is None:
        # Select hyperparameters
        if sel_model == "flexible_cnn":
            (hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters, poolsize,
             dropout_prob, batch_size) = hyperparameters_selector_2(model)
        elif sel_model == "cl_nn":
            hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters, poolsize = hyperparameters_selector(sel_model)
        else:
            hidden_size, num_layers, learning_rate, num_epochs, filtersize, num_filters, poolsize = hyperparameters_selector(sel_model)

    if gray:
        g = 1
    else:
        g = 3

    # Create the model, loss function, and optimizer
    train_loss_list = []
    train_accs_list = []
    start_time = time.time()

    # Initialize the plot for real-time visualization
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Hyperparameters: hidden_size={hidden_size}, num_layers={num_layers}, learning_rate={learning_rate}, num_epochs={num_epochs}", fontsize=16)

    # Setup loss plot
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    loss_line, = ax1.plot([], [], 'b-', label='Train Loss')
    ax1.legend()

    # Setup accuracy plot
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    acc_line, = ax2.plot([], [], 'orange', label='Train Accuracy')
    ax2.legend()

    # To avoid the plot from being garbage collected
    plt.show(block=False)

    if sel_model == "flexible_cnn":
        # Reshape data for FlexibleCNN
        train_images = train_images.view(-1, g, size, size)  # Reshape to (batch_size, channels, height, width)

        # Convert to float32 to match model weight type
        train_images = train_images.float()
        train_labels = train_labels.long()  # Ensure labels are long type

        # Use the specialized training function for FlexibleCNN
        print("Training the FlexibleCNN model...")
        model = train_flexible_cnn(
            data=train_images,
            labels=train_labels,
            input_channels=g,
            input_size=size,
            num_classes=num_classes,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            using_set=using_set
        )

        # Since we're using a different training method, we don't have the same metrics
        # We'll create dummy metrics for compatibility with visualization code
        train_loss_list = [0] * num_epochs
        train_accs_list = [0] * num_epochs

        # Get the actual metrics from the model's training history if available
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'callback_metrics'):
            metrics = model.trainer.callback_metrics
            if 'train_loss' in metrics:
                train_loss_list[-1] = metrics['train_loss'].item()
            if 'train_acc' in metrics:
                train_accs_list[-1] = metrics['train_acc'].item() * 100  # Convert to percentage
    else:
        # Create the model based on selection
        match sel_model:
            case "cl_nn":
                model = FCNNClassifier_linear(input_size, hidden_size, num_layers, num_classes)
            case "improved_nn":
                model = ImprovedFCNNClassifier(input_size, hidden_size, num_layers, num_classes)
            case "cc_nn":
                model = myCNNClassifier(g=g, filtersize=filtersize, poolsize=poolsize, input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout_prob=0.5, learning_rate=learning_rate, num_filters=num_filters)
                train_images = train_images.view(-1, g, size, size)  # Reshape to (batch_size, channels, height, width)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(" ")
        WIP = None
        if WIP is not None and input("Do you want to use a trainer for training? The trainer will use the number of epochs defined as a max number of epochs. (y/n): \n(WIP) Don't work yet").lower() == 'y':
            print("Using trainer...")
            # Create a PyTorch Lightning trainer
            trainer = L.Trainer(
                max_epochs=10,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
            )
            # Train the model
            trainer.fit(model, train_images.float(), train_labels)
        else:
            print("Training the model...")
            with alive_bar(num_epochs, force_tty=True, max_cols=270) as bar:
                for epoch in range(num_epochs):
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

                    # Update the plots
                    loss_line.set_data(range(len(train_loss_list)), train_loss_list)
                    acc_line.set_data(range(len(train_accs_list)), train_accs_list)

                    # Adjust the plot limits if needed
                    ax1.relim()
                    ax1.autoscale_view()
                    ax2.relim()
                    ax2.autoscale_view()

                    # Redraw the figure
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                    if (epoch + 1) % 10 == 0:
                        print(f'Epoch [{epoch + 1}/{num_epochs}], '
                              f'Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%')
                    bar()

    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds\n')

    # Final update of the plots
    loss_line.set_data(range(len(train_loss_list)), train_loss_list)
    acc_line.set_data(range(len(train_accs_list)), train_accs_list)
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw()

    # Save the plots
    if not os.path.exists(os.path.join(path, 'models/' + sel_model)):
        os.makedirs(os.path.join(path, 'models/' + sel_model))
    plt.savefig(os.path.join(path, 'models/' + sel_model, using_set + 'NN.png'))

    # Turn off interactive mode
    plt.ioff()
    plt.show()

    # Visualize the model architecture
    print("Visualizing the model architecture...")
    print(model)

    # Save the model
    if not os.path.exists(os.path.join(path, 'models/' + sel_model)):
        os.makedirs(os.path.join(path, 'models/' + sel_model))
    torch.save(model.state_dict(), os.path.join(path, 'models/' + sel_model, using_set + 'NN.pth'))
    return input_size, hidden_size, num_layers, num_classes, le, learning_rate, num_epochs, D_m, num_filters, filtersize, poolsize



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
        g = 1
    else:
        img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
        g = 3
    if img is None:
        print(f"Error loading image: {image_name}")
        return None

    # get crop values
    values = get_image_data(image_name)
    x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
    # crop and resize the image
    croped_img = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]
    croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)

    # Prepare data for prediction
    if model == "flexible_cnn":
        # For FlexibleCNN, we need the image in (channels, height, width) format
        if g == 1:
            # For grayscale, add channel dimension
            d_img_array = np.expand_dims(croped_img, axis=0)  # Add batch dimension
            d_img_array = np.expand_dims(d_img_array, axis=0)  # Add channel dimension
        else:
            # For color, transpose to (batch, channels, height, width)
            d_img_array = np.expand_dims(croped_img, axis=0)  # Add batch dimension
            d_img_array = np.transpose(d_img_array, (0, 3, 1, 2))  # Reorder to (batch, channels, height, width)
    else:
        # For other models, flatten the image
        d_img = croped_img.flatten()
        d_img_array = []
        d_img_array.append(d_img)
        d_img_array = np.array(d_img_array)

    end_extract = time.time()

    # Normalize data
    print("normalizing data...")
    d_img_array = np.array(d_img_array) / 255.0

    # Load family model
    if model == "flexible_cnn":
        NN_fam = FlexibleCNN(
            input_channels=g,
            input_size=size,
            num_classes=f_num_classes,
            learning_rate=f_learning_rate
        )
        # Centering data
        print("centering data...")
        if g == 1:
            # For grayscale
            d_img_c_f = d_img_array - f_D_m.reshape(1, size, size)
        else:
            # For color, reshape D_m to match image dimensions
            d_img_c_f = d_img_array.copy()
            for c in range(g):
                d_img_c_f[:, c, :, :] -= f_D_m.reshape(size, size, g)[:, :, c].reshape(1, size, size)
    else:
        NN_fam = FCNNClassifier_linear(f_input_size, f_hidden_size, f_num_layers, f_num_classes)
        # Centering data
        print("centering data...")
        d_img_c_f = d_img_array - f_D_m

    NN_fam.load_state_dict(torch.load(os.path.join(path, 'models/' + model, 'f_NN.pth')))
    NN_fam.eval()  # Set the model to evaluation mode

    # Load manufacturer model
    if model == "flexible_cnn":
        NN_man = FlexibleCNN(
            input_channels=g,
            input_size=size,
            num_classes=m_num_classes,
            learning_rate=m_learning_rate
        )
        # Centering data
        print("centering data...")
        if g == 1:
            # For grayscale
            d_img_c_m = d_img_array - m_D_m.reshape(1, size, size)
        else:
            # For color, reshape D_m to match image dimensions
            d_img_c_m = d_img_array.copy()
            for c in range(g):
                d_img_c_m[:, c, :, :] -= m_D_m.reshape(size, size, g)[:, :, c].reshape(1, size, size)
    else:
        NN_man = FCNNClassifier_linear(m_input_size, m_hidden_size, m_num_layers, m_num_classes)
        # Centering data
        print("centering data...")
        d_img_c_m = d_img_array - m_D_m.reshape(1, size, size)

    NN_man.load_state_dict(torch.load(os.path.join(path, 'models/' + model, 'm_NN.pth')))
    NN_man.eval()  # Set the model to evaluation mode

    start_prediction = time.time()
    # Get predictions
    if model == "flexible_cnn":
        predicted_m, confidence_percentages_m = NN_man.predict(torch.from_numpy(d_img_c_m).float())
        predicted_f, confidence_percentages_f = NN_fam.predict(torch.from_numpy(d_img_c_f).float())
    else:
        with torch.no_grad():
            outputs_f = NN_fam(torch.from_numpy(d_img_c_f).float())
            _,predicted_f = torch.max(outputs_f, 1)

            outputs_m = NN_man(torch.from_numpy(d_img_c_m).float())
            _, predicted_m = torch.max(outputs_m, 1)
    end_prediction = time.time()

    # Convert predicted labels back to original labels
    predicted_f = le_fam.inverse_transform(predicted_f.numpy())
    predicted_m = le_man.inverse_transform(predicted_m.numpy())

    # Display images and their predictions
    plt.figure(figsize=(15, 5))
    plt.imshow(croped_img, cmap='gray' if gray else None)
    if model == "flexible_cnn":
        plt.title(f'Predicted family: {predicted_f[0]} with a confidence of {round(confidence_percentages_f[0], 2)}\n Predicted manufacturer: {predicted_m[0]}  with a confidence of {round(confidence_percentages_m[0], 2)}')
    else:
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
    nn_test_s(data_family, label_family, "f_", model, f_D_m)
    print("\nTesting manufacturer model...")
    nn_test_s(data_manufacturer, label_manufacturer, "m_", model, m_D_m)



def nn_test_s(data, label, used_set, used_model, D_m):
    # load the models
    if used_set == "f_":
        input_size = f_input_size
        hidden_size = f_hidden_size
        num_layers = f_num_layers
        num_classes = f_num_classes
        le = le_fam
        learning_rate = f_learning_rate
        filtersize = f_filtersize
        num_filters = f_num_filter
        poolsize = f_poolsize
    else:
        input_size = m_input_size
        hidden_size = m_hidden_size
        num_layers = m_num_layers
        num_classes = m_num_classes
        le = le_man
        learning_rate = m_learning_rate
        filtersize = m_filtersize
        num_filters = m_num_filter
        poolsize = m_poolsize

    from data_extract import size, gray
    if gray:
        g = 1
    else:
        g = 3

    label = le.transform(label)

    # convert data to numpy array
    data = np.array(data)
    print("normalizing data...")
    data = data / 255.0

    print("centering data...")
    data = data - D_m
    label = np.array(label)

    # Convert entire datasets to tensors
    test_images = torch.from_numpy(data)
    test_labels = torch.from_numpy(label)

    # Create and load the appropriate model
    match used_model:
        case "cl_nn":
            model = FCNNClassifier_linear(input_size, hidden_size, num_layers, num_classes)
            model.load_state_dict(torch.load(os.path.join(path, 'models/' + used_model, used_set + 'NN.pth')))
        case "improved_nn":
            model = ImprovedFCNNClassifier(input_size, hidden_size, num_layers, num_classes)
            model.load_state_dict(torch.load(os.path.join(path, 'models/' + used_model, used_set + 'NN.pth')))
        case "cc_nn":
            model = CustomCNN(input_channels=g, num_filters=num_filters, filter_size=filtersize, pool_size=poolsize,
                              hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers,
                              learning_rate=learning_rate, input_image_size=size)
            model.load_state_dict(torch.load(os.path.join(path, 'models/' + used_model, used_set + 'NN.pth')))
            # Reshape for CNN
            test_images = test_images.view(-1, g, size, size)
        case "flexible_cnn":
            model = FlexibleCNN(
                input_channels=g,
                input_size=size,
                num_classes=num_classes,
                learning_rate=learning_rate,
            )
            model.load_state_dict(torch.load(os.path.join(path, 'models/' + used_model, used_set + 'NN.pth')))
            # Reshape for CNN
            test_images = test_images.view(-1, g, size, size)
            # Convert to float32 to match model weight type
            test_images = test_images.float()
            test_labels = test_labels.long()  # Ensure labels are long type

    model.eval()  # Set the model to evaluation mode

    # Predict using the model
    print("Predicting test data...")
    pred_s_time = time.time()
    test_outputs = model(test_images.float())
    pred_e_time = time.time()

    # Calculate test accuracy
    print("\nCalculating test accuracy...")
    accuracy_s_time = time.time()
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_accuracy = 100 * (test_predicted == test_labels).sum().item() / len(test_labels)
    # calculate partial accuracy for each individual classes/labels
    test_predicted = le.inverse_transform(test_predicted.numpy()) # tensor to ndarray using .numpy()
    test_labels = le.inverse_transform(test_labels.numpy())


    for target_label in set(test_labels):
        # Filtrer les indices où l'étiquette réelle correspond à "Boeing"
        # for i, label in enumerate(test_labels)
        #   if label == target_label:
        #       target_indices.append(i)
        target_indices = [i for i, label in enumerate(test_labels) if label == target_label]

        # Extraire les prédictions et étiquettes réelles pour ces indices
        filtered_predictions = [test_predicted[i] for i in target_indices]
        filtered_true_labels = [test_labels[i] for i in target_indices]

        # Calculer l'exactitude pour l'étiquette cible
        accuracy = accuracy_score(filtered_true_labels, filtered_predictions)
        print(f"accuracy for {target_label}: {accuracy * 100:.2f}%")

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
        print("\nThe two models (family/manufacturer) have the same parameters")
        print("learning rate: " + str(f_learning_rate))
        print("hidden size: " + str(f_hidden_size))
        print("number of layers: " + str(f_num_layers))
        print("number of epochs: " + str(f_num_epochs))
        print("number of filters: " + str(f_num_filter))
        print("filtersize: " + str(f_filtersize))
        print("poolsize: " + str(f_poolsize))
    else:
        print(" ")
        print("Family model parameters:")
        print("learning rate: " + str(f_learning_rate))
        print("hidden size: " + str(f_hidden_size))
        print("number of layers: " + str(f_num_layers))
        print("number of epochs: " + str(f_num_epochs))
        print("number of filters: " + str(f_num_filter))
        print("filtersize: " + str(f_filtersize))
        print("poolsize: " + str(f_poolsize))

        print(" ")
        print("Manufacturer model parameters:")
        print("learning rate: " + str(m_learning_rate))
        print("hidden size: " + str(m_hidden_size))
        print("number of layers: " + str(m_num_layers))
        print("number of epochs: " + str(m_num_epochs))
        print("number of filters: " + str(m_num_filter))
        print("filtersize: " + str(m_filtersize))
        print("poolsize: " + str(f_poolsize))

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

## Copilot improved class
class ImprovedFCNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(ImprovedFCNNClassifier, self).__init__()

        # Create a list to hold all layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.BatchNorm1d(hidden_size))
        self.layers.append(nn.LeakyReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout_prob))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers[:-1]:
            x = layer(x)  # Apply all layers except the last one
        x = self.layers[-1](x)  # Apply the final layer (output layer)
        return x

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


## based from a book: The StatQuest Illustrated Guide to Neural Networks and AI: With hands-on examples in PyTorch!!!
# TODO: ameliorate and devlopp this new NN
class myCNNClassifier(L.LightningModule):

    def __init__(self, g=1, filtersize=3, poolsize=2, input_size=4, hidden_size=1, num_classes=2, num_layers=1, dropout_prob=0.5, learning_rate=0.001, num_filters=32):
        super().__init__()  ## We call the __init__() for the parent, LightningModule, so that it
        ## can initialize itself as well.

        self.learning_rate = learning_rate

        ############################################################################
        ##
        ## Here is where we initialize the Weights and Biases for the CNN
        ##
        ############################################################################

        ## The filter is created and applied by nn.Conv2d().
        ## in_channels - The number of color channels that
        ##    the image has. Our black and white image only
        ##    has one channel. However, color pictures usually have 3.
        ## out_channels - If we had multiple input channels, we could merge
        ##    them down to one output. Or we can increase the number of
        ##    output channels if we want.
        ## kernel_size - The size of the filter (aka kernel). In this case
        ##    we want a 3x3 filter, but you can select all kinds of sizes,
        ##    including sizes that are more rectangular than square.

        #self.conv = nn.Conv2d(in_channels=g, out_channels=g, kernel_size=filtersize)

        ## nn.MaxPool2d() does the max pooling step.
        ## kernel_size - The size of the filter (aka kernel) that does the
        ##    max pooling. We're using a 2x2 grid for our filter.
        ## stride - How much to move the filter each step. In this case
        ##    we're moving it 2 units. Thus, our 2x2 filter does max pooling
        ##    before moving 2 units over (or down). This means that our
        ##    max pooling filter never overlaps itself.

        #self.pool = nn.MaxPool2d(kernel_size=poolsize, stride=2)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = g, out_channels=num_filters, kernel_size=filtersize, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=poolsize, stride=2),
            nn.Dropout(dropout_prob)
        )
        # Calculer la taille après les couches convolutives
        conv_output_size = (input_size // poolsize) * (input_size // poolsize) * num_filters

        ## Lastly, we create the "normal" neural network that has
        ## 4 inputs, in_features=4, going to a single activation function, out_features=1,
        ## in a single hidden layer...
        # Couches entièrement connectées
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(conv_output_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout_prob))
        self.layers.append(nn.Linear(hidden_size, num_classes))

        # Initialize weights
        self._initialize_weights()

        ## We'll use Cross Entropy to calculate the loss between what the
        ## neural network's predictions and actual, or known, species for
        ## each row in the dataset.
        ## To learn more about Cross Entropy, see: https://youtu.be/6ArSys5qHAU
        ## NOTE: nn.CrossEntropyLoss applies a SoftMax function to the values
        ## we give it, so we don't have to do that oursevles. However,
        ## when we use this neural network (after it has been trained), we'll
        ## have to remember to apply a SoftMax function to the output.
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        ## First we apply a filter to the input image
        x = self.conv_layers(x)

        ## Then we run the output from the filter through a ReLU...
        #x = F.relu(x)

        ## Then we run the output from the ReLU through a Max Pooling layer...
        #x = self.pool(x)

        ## Now, at this point we have a square matrix of values.
        ## So, in order to use those values as inputs to
        ## a neural network, we use torch.flatten() to
        ## turn the matrix into a vector.
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        ## Now we run the flattened values through a neural network
        ## with a single hidden layer and a single ReLU activation
        ## function in that layer.
        #x = self.input_to_hidden(x)
        #x = F.relu(x)
        #x = self.hidden_to_output(x)

        # Pass through fully connected layers
        for layer in self.layers:
            x = layer(x)

        return x

    def configure_optimizers(self):
        ## In this example, configuring the optimizer
        ## consists of passing it the weights and biases we want
        ## to optimize, which are all in self.parameters(),
        ## and setting the learning rate with lr=0.001.
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        ## The first thing we do is split 'batch'
        ## into the input and label values.
        inputs, labels = batch

        ## Then we run the input through the neural network
        outputs = self.forward(inputs)

        ## Then we calculate the loss.
        loss = self.loss(outputs, labels)

        ## Lastly, we could add the loss to a log file
        ## so that we can graph it later. This would
        ## help us decide if we have done enough training
        ## Ideally, if we do enough training, the loss
        ## should be small and not getting any smaller.
        self.log("loss", loss)

        return loss

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class CustomCNN(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_filters=32,
                 filter_size=3,
                 pool_size=2,
                 hidden_size=128,
                 num_classes=10,
                 num_layers=2,
                 dropout_prob=0.5,
                 learning_rate=0.001,
                 input_image_size=64):
        super(CustomCNN, self).__init__()

        self.learning_rate = learning_rate

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=filter_size, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=2),
            nn.Dropout(dropout_prob)
        )

        # Calculer dynamiquement la taille de sortie après les couches convolutives
        self.conv_output_size = self._get_conv_output_size(input_image_size, input_channels, num_filters, filter_size,
                                                           pool_size)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.conv_output_size, hidden_size))
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.fc_layers.append(nn.BatchNorm1d(hidden_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_prob))
        self.fc_layers.append(nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        # Passer par les couches convolutives
        #x = self.conv_layers(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, 1)  # Aplatir sauf la dimension batch
        # Passer par les couches entièrement connectées
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def _get_conv_output_size(self, input_size, input_channels, num_filters, filter_size, pool_size):
        # Simule un passage à travers les couches convolutives pour calculer la taille de sortie
        dummy_input = torch.zeros(1, input_channels, input_size, input_size)
        output = self.conv_layers(dummy_input)
        return int(torch.flatten(output, 1).shape[1])
