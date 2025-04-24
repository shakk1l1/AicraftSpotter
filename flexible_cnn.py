import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
from path import path


class PlotMetricsCallback(Callback):
    """
    A callback to plot training metrics in real-time during training.
    """

    def __init__(self, model_name, model_type):
        super().__init__()
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.model_name = model_name
        self.model_type = model_type
        self.fig = None
        self.loss_line = None
        self.acc_line = None
        self.val_loss_line = None
        self.val_acc_line = None
        self.ax1 = None
        self.ax2 = None

    def on_fit_start(self, trainer, pl_module):
        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle(f"Training Metrics - {self.model_name}", fontsize=16)

        # Setup loss plot
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training and Validation Loss')
        self.loss_line, = self.ax1.plot([], [], 'b-', label='Train Loss')
        self.val_loss_line, = self.ax1.plot([], [], 'r-', label='Val Loss')
        self.ax1.legend()

        # Setup accuracy plot
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.set_title('Training and Validation Accuracy')
        self.acc_line, = self.ax2.plot([], [], 'orange', label='Train Accuracy')
        self.val_acc_line, = self.ax2.plot([], [], 'green', label='Val Accuracy')
        self.ax2.legend()

        # Show the plot
        plt.show(block=False)

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the metrics
        metrics = trainer.callback_metrics

        # Extract and store the metrics
        if 'train_loss' in metrics:
            self.train_losses.append(metrics['train_loss'].item())
        if 'train_acc' in metrics:
            self.train_accs.append(metrics['train_acc'].item() * 100)  # Convert to percentage

        # Update the plots
        self._update_plots()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the metrics
        metrics = trainer.callback_metrics

        # Extract and store the metrics
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'].item())
        if 'val_acc' in metrics:
            self.val_accs.append(metrics['val_acc'].item() * 100)  # Convert to percentage

        # Update the plots
        self._update_plots()

    def _update_plots(self):
        # Update the loss plot
        self.loss_line.set_data(range(len(self.train_losses)), self.train_losses)
        if self.val_losses:
            self.val_loss_line.set_data(range(len(self.val_losses)), self.val_losses)

        # Update the accuracy plot
        self.acc_line.set_data(range(len(self.train_accs)), self.train_accs)
        if self.val_accs:
            self.val_acc_line.set_data(range(len(self.val_accs)), self.val_accs)

        # Adjust the plot limits
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_fit_end(self, trainer, pl_module):
        # Final update of the plots
        self._update_plots()

        # Save the plots
        if not os.path.exists(os.path.join(path, 'models/' + self.model_type)):
            os.makedirs(os.path.join(path, 'models/' + self.model_type))
        plt.savefig(os.path.join(path, 'models/' + self.model_type, self.model_name + '_NN.png'))

        # Turn off interactive mode
        plt.ioff()
        plt.show()


class ResidualBlock(nn.Module):
    """
    A residual block with skip connections to help with gradient flow in deep networks.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection handling different dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FlexibleCNN(L.LightningModule):
    """
    A flexible CNN model that can handle variable input sizes, color channels, and output classes.
    Designed to be as accurate as possible for labeling tasks.
    """

    def __init__(self,
                 input_channels=3,
                 input_size=64,
                 num_classes=10,
                 learning_rate=0.001,
                 dropout_prob=0.3,
                 base_filters=64):
        super(FlexibleCNN, self).__init__()

        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, 2, stride=2)

        # Calculate the size after all convolutional and pooling layers
        # Initial size reduction from conv1 and maxpool
        size_after_initial = input_size // 4
        # Further reduction from the three residual layers
        size_after_res = size_after_initial // 4

        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(base_filters * 4, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block may change dimensions
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global average pooling and final classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def predict(self, x):
        """
        Make a prediction with confidence percentage.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            tuple: (predicted_class, confidence_percentage)
                - predicted_class: The predicted class index
                - confidence_percentage: The confidence percentage for the prediction (0-100)
        """
        # Ensure the model is in evaluation mode
        self.eval()

        # Forward pass
        with torch.no_grad():
            outputs = self(x)

            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)

            # Get the predicted class and its probability
            confidence, predicted_class = torch.max(probabilities, dim=1)

            # Convert to percentage (0-100)
            confidence_percentage = confidence * 100

        return predicted_class, confidence_percentage


def train_flexible_cnn(data, labels, input_channels=3, input_size=64, num_classes=None,
                       batch_size=32, learning_rate=0.001, num_epochs=50, validation_split=0.2, using_set=None):
    """
    Train the FlexibleCNN model with the given data and labels.

    Args:
        data: Tensor of shape (num_images, num_channels, height, width)
        labels: Tensor of labels
        input_channels: Number of input channels (default: 3)
        input_size: Size of input images (default: 64)
        num_classes: Number of output classes (default: inferred from labels)
        batch_size: Batch size for training (default: 32)
        learning_rate: Learning rate (default: 0.001)
        num_epochs: Number of training epochs (default: 50)
        validation_split: Fraction of data to use for validation (default: 0.2)
        using_set: String identifier for the model (e.g., 'f_' for family, 'm_' for manufacturer)
                  Used for naming the saved plots (default: None)

    Returns:
        Trained FlexibleCNN model
    """
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader, random_split

    # Determine number of classes if not provided
    if num_classes is None:
        num_classes = len(torch.unique(labels))

    # Create dataset and split into train/validation
    dataset = TensorDataset(data, labels)
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = FlexibleCNN(
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        learning_rate=learning_rate
    )

    # Create trainer with plot metrics callback
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(monitor='val_loss', save_top_k=1),
            PlotMetricsCallback(model_name=using_set if using_set is not None else 'model', model_type='flexible_cnn')
        ]
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    return model
