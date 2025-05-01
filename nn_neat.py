import os
import cv2
import joblib
import time
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import neat as neat_python
import pickle
from path import path
from database import get_image_data

# Global variables
f_input_size, f_num_classes, f_num_generations = None, None, None
m_input_size, m_num_classes, m_num_generations = None, None, None
le_fam = None
le_man = None
f_D_m, m_D_m = None, None
f_winner_net, m_winner_net = None, None
f_config, m_config = None, None

def load_neat_models(model):
    """
    Load the NEAT models
    :param model: model name
    :return: None
    """
    # import the functions from data_extract
    # to avoid circular imports
    from data_extract import change_size
    global f_input_size, f_num_classes, f_num_generations
    global m_input_size, m_num_classes, m_num_generations
    global le_fam, le_man
    global f_D_m, m_D_m
    global f_winner_net, m_winner_net
    global f_config, m_config

    # Check if the model files exist
    if os.path.exists(os.path.join(path, 'models/' + model, 'm_winner.pkl')) and os.path.exists(os.path.join(path, 'models/' + model, 'f_winner.pkl')):
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
    f_num_classes = joblib.load(os.path.join(path, 'models/' + model, 'f_num_classes.pkl'))
    f_num_generations = joblib.load(os.path.join(path, 'models/' + model, 'f_num_generations.pkl'))

    m_input_size = joblib.load(os.path.join(path, 'models/' + model, 'm_input_size.pkl'))
    m_num_classes = joblib.load(os.path.join(path, 'models/' + model, 'm_num_classes.pkl'))
    m_num_generations = joblib.load(os.path.join(path, 'models/' + model, 'm_num_generations.pkl'))

    f_D_m = joblib.load(os.path.join(path, 'models/' + model, 'f_D_m.pkl'))
    m_D_m = joblib.load(os.path.join(path, 'models/' + model, 'm_D_m.pkl'))

    # Load the winner networks
    with open(os.path.join(path, 'models/' + model, 'f_winner.pkl'), 'rb') as f:
        f_winner_net = pickle.load(f)

    with open(os.path.join(path, 'models/' + model, 'm_winner.pkl'), 'rb') as f:
        m_winner_net = pickle.load(f)

    # Load the configurations
    f_config = neat_python.config.Config(
        neat_python.DefaultGenome,
        neat_python.DefaultReproduction,
        neat_python.DefaultSpeciesSet,
        neat_python.DefaultStagnation,
        os.path.join(path, 'models/' + model, 'f_config.txt')
    )

    m_config = neat_python.config.Config(
        neat_python.DefaultGenome,
        neat_python.DefaultReproduction,
        neat_python.DefaultSpeciesSet,
        neat_python.DefaultStagnation,
        os.path.join(path, 'models/' + model, 'm_config.txt')
    )

    # call the function to change size and gray
    change_size(model)
    print("Models loaded")

def create_neat_config(input_size, output_size, filename):
    """
    Create a NEAT configuration file
    :param input_size: number of inputs
    :param output_size: number of outputs
    :param filename: filename to save the configuration
    :return: None
    """
    config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 0.95
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.1
activation_options      = sigmoid tanh relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = partial_direct 0.5

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = {input_size}
num_outputs             = {output_size}

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""

    with open(filename, 'w') as f:
        f.write(config_text)

def eval_genomes(genomes, config, X, y):
    """
    Evaluate the fitness of each genome
    :param genomes: list of genomes to evaluate
    :param config: NEAT configuration
    :param X: input data
    :param y: target labels
    :return: None
    """
    for genome_id, genome in genomes:
        net = neat_python.nn.FeedForwardNetwork.create(genome, config)

        # Initialize fitness
        genome.fitness = 0

        # Evaluate the genome on all samples
        correct_predictions = 0
        for xi, yi in zip(X, y):
            # Get the network output
            output = net.activate(xi)

            # Get the predicted class (index of max value)
            predicted_class = np.argmax(output)

            # Check if prediction is correct
            if predicted_class == yi:
                correct_predictions += 1

        # Set fitness as accuracy
        genome.fitness = correct_predictions / len(y)

def neat_train(data_family, label_family, data_manufacturer, label_manufacturer, model):
    """
    Train the NEAT models for family and manufacturer classification
    :param data_family: family data
    :param label_family: family labels
    :param data_manufacturer: manufacturer data
    :param label_manufacturer: manufacturer labels
    :param model: model name
    :return: None
    """
    global f_input_size, f_num_classes, f_num_generations
    global m_input_size, m_num_classes, m_num_generations
    global le_fam, le_man
    global f_D_m, m_D_m
    global f_winner_net, m_winner_net
    global f_config, m_config

    from data_extract import size

    # Get number of generations
    num_generations = int(input("\nNumber of generations to evolve (recommended: 50-200): "))
    f_num_generations = num_generations
    m_num_generations = num_generations

    print("\nTraining family model...")
    f_input_size, f_num_classes, le_fam, f_D_m, f_winner_net, f_config = neat_train_s(
        data_family, label_family, model, size, "f_", num_generations)

    print("\nTraining manufacturer model...")
    m_input_size, m_num_classes, le_man, m_D_m, m_winner_net, m_config = neat_train_s(
        data_manufacturer, label_manufacturer, model, size, "m_", num_generations)

    # Create directory if it doesn't exist
    if not os.path.exists(os.path.join(path, 'models/' + model)):
        os.makedirs(os.path.join(path, 'models/' + model))

    # Save the models
    joblib.dump(le_fam, os.path.join(path, 'models/' + model, 'le_fam.pkl'))
    joblib.dump(le_man, os.path.join(path, 'models/' + model, 'le_man.pkl'))

    joblib.dump(f_input_size, os.path.join(path, 'models/' + model, 'f_input_size.pkl'))
    joblib.dump(f_num_classes, os.path.join(path, 'models/' + model, 'f_num_classes.pkl'))
    joblib.dump(f_num_generations, os.path.join(path, 'models/' + model, 'f_num_generations.pkl'))

    joblib.dump(m_input_size, os.path.join(path, 'models/' + model, 'm_input_size.pkl'))
    joblib.dump(m_num_classes, os.path.join(path, 'models/' + model, 'm_num_classes.pkl'))
    joblib.dump(m_num_generations, os.path.join(path, 'models/' + model, 'm_num_generations.pkl'))

    # Save the mean values
    joblib.dump(f_D_m, os.path.join(path, 'models/' + model, 'f_D_m.pkl'))
    joblib.dump(m_D_m, os.path.join(path, 'models/' + model, 'm_D_m.pkl'))

    # Save the winner networks
    with open(os.path.join(path, 'models/' + model, 'f_winner.pkl'), 'wb') as f:
        pickle.dump(f_winner_net, f)

    with open(os.path.join(path, 'models/' + model, 'm_winner.pkl'), 'wb') as f:
        pickle.dump(m_winner_net, f)

    # Save the configurations
    f_config_path = os.path.join(path, 'models/' + model, 'f_config.txt')
    m_config_path = os.path.join(path, 'models/' + model, 'm_config.txt')

    # Copy the config files to the model directory
    import shutil
    shutil.copy(os.path.join(path, 'f_config.txt'), f_config_path)
    shutil.copy(os.path.join(path, 'm_config.txt'), m_config_path)

    # import last version of size and gray
    from data_extract import size, gray
    joblib.dump(size, os.path.join(path, 'models/' + model, 'size.pkl'))
    joblib.dump(gray, os.path.join(path, 'models/' + model, 'gray.pkl'))

    print("Models saved")

def neat_train_s(data, label, model, size, using_set, num_generations):
    """
    Train a single NEAT model
    :param data: training data
    :param label: training labels
    :param model: model name
    :param size: image size
    :param using_set: prefix for model files (f_ or m_)
    :param num_generations: number of generations to evolve
    :return: input_size, num_classes, le, D_m, winner_net, config
    """
    # Convert data to numpy array
    print("Converting data to numpy arrays...")
    data = np.array(data) / 255.0

    # Calculate mean and center data
    print("Calculating mean...")
    D_m = np.mean(data, axis=0)
    print("Centering data...")
    data = data - D_m

    # Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)

    # Get input and output sizes
    input_size = data.shape[1]  # Flattened image size
    num_classes = len(le.classes_)

    # Create config file
    config_path = os.path.join(path, f'{using_set}config.txt')
    create_neat_config(input_size, num_classes, config_path)

    # Load config
    config = neat_python.config.Config(
        neat_python.DefaultGenome,
        neat_python.DefaultReproduction,
        neat_python.DefaultSpeciesSet,
        neat_python.DefaultStagnation,
        config_path
    )

    # Create the population
    print("Creating initial population...")
    p = neat_python.Population(config)

    # Add reporters
    p.add_reporter(neat_python.StdOutReporter(True))
    stats = neat_python.StatisticsReporter()
    p.add_reporter(stats)

    # Create a checkpoint every 10 generations
    p.add_reporter(neat_python.Checkpointer(10, filename_prefix=os.path.join(path, f'neat-checkpoint-{using_set}')))

    # Create a custom reporter for plotting
    class PlotReporter(neat_python.reporting.BaseReporter):
        def __init__(self):
            self.generation = 0
            self.best_fitness = []
            self.avg_fitness = []
            self.generations = []

            # Initialize the plot
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.fig.suptitle(f"NEAT Training Progress - {using_set}", fontsize=16)

            # Setup fitness plot
            self.ax.set_xlabel('Generation')
            self.ax.set_ylabel('Fitness')
            self.ax.set_title('Fitness over Generations')
            self.best_line, = self.ax.plot([], [], 'b-', label='Best Fitness')
            self.avg_line, = self.ax.plot([], [], 'r-', label='Average Fitness')
            self.ax.legend()

            # Show the plot
            plt.show(block=False)

        def end_generation(self, config, population, species_set):
            self.generation += 1
            self.generations.append(self.generation)

            best_genome = None
            best_fitness = -1
            for _, genome in population.items():
                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_genome = genome

            self.best_fitness.append(best_fitness)

            # Calculate average fitness
            avg_fitness = sum(g.fitness for g in population.values()) / len(population)
            self.avg_fitness.append(avg_fitness)

            # Update the plot
            self.best_line.set_data(self.generations, self.best_fitness)
            self.avg_line.set_data(self.generations, self.avg_fitness)

            # Adjust the plot limits
            self.ax.relim()
            self.ax.autoscale_view()

            # Redraw the figure
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        def found_solution(self, config, generation, best):
            pass

        def save_plot(self, filename):
            plt.savefig(filename)

    plot_reporter = PlotReporter()
    p.add_reporter(plot_reporter)

    # Run the evolution
    print(f"Evolving for {num_generations} generations...")
    start_time = time.time()

    # Create a partial function for evaluation
    eval_function = lambda genomes, config: eval_genomes(genomes, config, data, encoded_labels)

    # Run the evolution with a progress bar
    with alive_bar(num_generations, force_tty=True, title=f'Evolving {using_set} NEAT model') as bar:
        winner = p.run(eval_function, num_generations, step_callback=lambda _: bar())

    end_time = time.time()
    print(f"Evolution completed in {end_time - start_time:.2f} seconds")

    # Save the final plot
    if not os.path.exists(os.path.join(path, 'models/' + model)):
        os.makedirs(os.path.join(path, 'models/' + model))
    plot_reporter.save_plot(os.path.join(path, 'models/' + model, f'{using_set}NEAT.png'))

    # Create the winner network
    winner_net = neat_python.nn.FeedForwardNetwork.create(winner, config)

    # Visualize the winner network
    if input("Visualize the winner network? (y/n) ").lower() == 'y':
        try:
            import graphviz
            dot = neat_python.visualize.draw_net(config, winner, view=True)
            dot.render(os.path.join(path, 'models/' + model, f'{using_set}winner-network'))
        except ImportError:
            print("Graphviz not installed, skipping visualization")

    # Return the necessary values
    return input_size, num_classes, le, D_m, winner_net, config

def neat_predict(image_name):
    """
    Predict the family and manufacturer of an image using NEAT
    :param image_name: name of the image
    :return: None
    """
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

    print("\nPredicting family...")

    print("Normalizing data...")
    d_img_array = np.array(d_img_array) / 255.0
    # centering
    print("Centering data...")
    start_center_f = time.time()
    d_img_c = d_img_array - f_D_m
    end_center_f = time.time()

    # Predicting
    print("\nPredicting...")
    start_predict_f = time.time()

    # Get the network output
    output_f = f_winner_net.activate(d_img_c[0])

    # Get the predicted class and confidence
    predicted_class_f = np.argmax(output_f)
    confidence_f = output_f[predicted_class_f] * 100  # Convert to percentage

    # Convert to original label
    predicted_f = le_fam.inverse_transform([predicted_class_f])[0]

    end_predict_f = time.time()
    print(f"Predicted family: {predicted_f} ({confidence_f:.2f}%)")

    print("\nPredicting manufacturer...")
    # centering
    start_center_m = time.time()
    d_img_c = d_img_array - m_D_m
    end_center_m = time.time()

    # Predicting
    print("\nPredicting...")
    start_predict_m = time.time()

    # Get the network output
    output_m = m_winner_net.activate(d_img_c[0])

    # Get the predicted class and confidence
    predicted_class_m = np.argmax(output_m)
    confidence_m = output_m[predicted_class_m] * 100  # Convert to percentage

    # Convert to original label
    predicted_m = le_man.inverse_transform([predicted_class_m])[0]

    end_predict_m = time.time()
    print(f"Predicted manufacturer: {predicted_m} ({confidence_m:.2f}%)")

    # Display the image and predictions
    plt.figure(figsize=(10, 6))
    plt.imshow(croped_img, cmap='gray' if gray else None)
    plt.title(f'Predicted family: {predicted_f} ({confidence_f:.2f}%)\nPredicted manufacturer: {predicted_m} ({confidence_m:.2f}%)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # print the time taken for each step
    print(f"\nExtracting time: {end_extract - start_extract:.2f} seconds")
    print(f"\nFamily centering time: {end_center_f - start_center_f:.2f} seconds")
    print(f"Family predicting time: {end_predict_f - start_predict_f:.2f} seconds")
    print(f"\nManufacturer centering time: {end_center_m - start_center_m:.2f} seconds")
    print(f"Manufacturer predicting time: {end_predict_m - start_predict_m:.2f} seconds")
    print(f"\nTotal predicting time: {end_predict_m - start_predict:.2f} seconds")
    return None

def neat_test(data_family, label_family, data_manufacturer, label_manufacturer):
    """
    Test the NEAT models for family and manufacturer
    :param data_family: family data
    :param label_family: family labels
    :param data_manufacturer: manufacturer data
    :param label_manufacturer: manufacturer labels
    :return: None
    """
    print("\nTesting family model...")
    neat_test_s(data_family, label_family, "f_", f_D_m, f_winner_net, le_fam)
    print("\nTesting manufacturer model...")
    neat_test_s(data_manufacturer, label_manufacturer, "m_", m_D_m, m_winner_net, le_man)

def neat_test_s(data, label, used_set, D_m, winner_net, le):
    """
    Test a single NEAT model
    :param data: test data
    :param label: test labels
    :param used_set: prefix for model files (f_ or m_)
    :param D_m: mean value for centering
    :param winner_net: trained NEAT network
    :param le: label encoder
    :return: None
    """
    # convert data to numpy array
    data = np.array(data)
    print("Normalizing data...")
    data = data / 255.0

    print("Centering data...")
    start_center = time.time()
    data = data - D_m
    end_center = time.time()

    # Encode labels
    encoded_labels = le.transform(label)

    print("\nPredicting and calculating score...")
    start_predict = time.time()

    # Make predictions
    predictions = []
    with alive_bar(len(data), force_tty=True, title=f'Testing {used_set} NEAT model') as bar:
        for i, sample in enumerate(data):
            # Get the network output
            output = winner_net.activate(sample)

            # Get the predicted class
            predicted_class = np.argmax(output)
            predictions.append(predicted_class)
            bar()

    end_predict = time.time()

    # Calculate accuracy
    accuracy = accuracy_score(encoded_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Calculate per-class accuracy
    for target_label in set(label):
        # Get indices where the true label is the target
        target_indices = [i for i, l in enumerate(label) if l == target_label]

        # Get predictions and true labels for these indices
        filtered_predictions = [le.inverse_transform([predictions[i]])[0] for i in target_indices]
        filtered_true_labels = [label[i] for i in target_indices]

        # Calculate accuracy for this class
        class_accuracy = accuracy_score(filtered_true_labels, filtered_predictions)
        print(f"Accuracy for {target_label}: {class_accuracy * 100:.2f}%")

    # Print timing information
    print(f"\nCentering time: {end_center - start_center:.2f} seconds")
    print(f"Prediction time: {end_predict - start_predict:.2f} seconds")
    print(f"Total testing time: {end_predict - start_center:.2f} seconds")

def neat_param(model):
    """
    Show the parameters of the NEAT models
    :param model: model name
    :return: None
    """
    from data_extract import size, gray
    print(" ")
    print("Showing model parameters")
    print("Size of the image: " + str(size))
    print("Gray scale: " + str(gray))

    print("\nFamily model parameters:")
    print(f"Input size: {f_input_size}")
    print(f"Number of classes: {f_num_classes}")
    print(f"Number of generations: {f_num_generations}")

    # Get network structure information
    f_nodes = len(f_winner_net.node_evals)
    f_connections = sum(1 for node, links in f_winner_net.node_evals for link in links)
    print(f"Network nodes: {f_nodes}")
    print(f"Network connections: {f_connections}")

    print("\nManufacturer model parameters:")
    print(f"Input size: {m_input_size}")
    print(f"Number of classes: {m_num_classes}")
    print(f"Number of generations: {m_num_generations}")

    # Get network structure information
    m_nodes = len(m_winner_net.node_evals)
    m_connections = sum(1 for node, links in m_winner_net.node_evals for link in links)
    print(f"Network nodes: {m_nodes}")
    print(f"Network connections: {m_connections}")
