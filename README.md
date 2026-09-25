# Aircraft Spotter

A Python course project for identifying an aircraft's **manufacturer** and
**family** from photographs. It explores classical machine-learning approaches
and neural networks through an interactive command-line interface.

Developed for the MA1 Data Driven Engineering course. **Primary implementation:
Thibault Fesler.** The original project credits Thibault Fesler, Sophie Michiels,
Emanuel Cotan, and Ilyas Fatih.

## What is included

- Image loading, bounding-box cropping, resizing, and optional greyscale conversion.
- PCA and support-vector classifiers, ridge-based methods, and experimental
  regression approaches.
- Neural-network implementations, including a configurable convolutional network.
- Commands for training, loading saved models, testing, inspecting parameters,
  and predicting an individual image's manufacturer and family.
- A SQLite database for image bounding-box information and saved model files.

The model menu distinguishes experimental regression methods and work-in-progress
options. Their presence in the menu does not imply equivalent classification
performance or completeness.

## Setup

The source uses Python 3.10+ syntax. Dependencies are listed in
[`requirements.txt`](requirements.txt); a fully pinned environment is not provided.

```sh
git clone https://github.com/shakk1l1/AicraftSpotter.git
cd AicraftSpotter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The repository includes a large image dataset, so downloading it can take time.

On the first launch, enter the absolute path to this repository's `data` folder
when prompted. The program saves that setting in the ignored `path.py` file.
Run the program from the repository root.

## Data layout

The course dataset contains 10,000 aircraft photographs. The application expects:

```text
data/
  images/                         # <image_id>.jpg
  images_box.txt                  # Bounding-box coordinates
  images_family_trainval.txt
  images_family_test.txt
  images_manufacturer_trainval.txt
  images_manufacturer_test.txt
  models/                         # Created when models are saved
```

The dataset also includes train/validation subdivisions and variant labels.
The main data-loading paths use `trainval` for training and `test` for evaluation;
manufacturer and family are the application's prediction targets.

## Using the interface

Type `help` to see the available commands. A typical interactive sequence is:

1. Enter `model`, then select a model such as `svc`.
2. Enter `train` and answer the preprocessing and model-parameter prompts.
3. Enter `test` to evaluate the trained model on the test split.
4. Use `predict` for a dataset image or `param` to inspect model parameters.
5. Enter `esc` to exit.

Other commands include `load`, `path`, `show`, and `close`. Loading a model
requires a compatible saved model for the selected method. The exact prompts
depend on the method; training may require substantial time and memory.

## Code guide

| File | Purpose |
| --- | --- |
| `main.py`, `command_general.py` | Application startup and command handling |
| `welcome_script.py` | First-run data-path configuration |
| `data_extract.py`, `image_handler.py` | Dataset preparation and image handling |
| `database.py` | Bounding-box database access |
| `PCA_SVC.py` | PCA and support-vector methods |
| `lreg.py`, `cv.py` | Regression and cross-validation experiments |
| `nn.py`, `flexible_cnn.py` | Neural-network implementations |

## Evaluation and limitations

The project provides evaluation routines, but this README does not claim a
reproduced benchmark score. Report results together with the model, preprocessing,
training configuration, and evaluation split. Keep the test split separate from
model selection and hyperparameter tuning.

This is an academic exploration. Some menu options remain experimental, and
dependency versions and complete end-to-end execution have not been validated
as part of this documentation update.

## License

See [`LICENSE`](LICENSE) for the repository's license. Retain the project authors'
credits and consult the dataset's own documentation for image usage terms.
