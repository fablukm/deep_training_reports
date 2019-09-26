# keras_reports

## Why
When training Neural networks, we compare differnet choices of hyperparameters and architectures to determine which model performs best. To compare the different results, metrics need to be written down and put into a table by hand for presentation.

This package implements a practical all-in-one solution for _keras_ models. It trains the models with their respective configuration and writes a `.pdf` directly, which contains all information, history plots, a table comparing metrics of all the different models at once, a model summary for all used architectures, etc.

Training accuracy, loss, and learning rate plots are written in `tikz` and passed to LaTeX template.

The `.pdf` is rendered from a LaTeX template using Jinja2. This means that the template is __fully customisable__ to fit into conference `.tex` templates, or corporate report layouts.

## Getting started
1. Define your _keras_ models in `models.py`.
   Write each model configuration to a `.json` in the folder `model_configs`.
   Define the number of epochs, the batch size, optimizer, its parameters, a custom name for the study, and even add a link to the paper or github-repo containing the model.
2. Run `train.py`. This will do the following:
    1. Read all the configuration `.json` inside the folder `model_configs`,
    2. Train all models with parameters specified in the `.json`,
    3. After each training, write another `.json` with all training results to the directory `training_logs`,
    4. Run `report.py` to read each of the `training_logs/*.json`, and fill them into the LaTeX template `latex/template.tex`. This will render the template to a `.tex` file and call `pdflatex` to create a `.pdf`.

You can change the output directories, the document name and authors in the config file `report_config.json`.

__If you do not want to generate a summary of all sudies saved in `training_logs`__, you can specify a list of filenames in `report_config.json`. The report generator will only consider files passed in this list.

## Configuration
All configurations are `.json`, as the best available compromise between human and machine readability.
### Configuration of the reporting tool `report_config.json`
The following keys are specified therein:

| Key | Default | Description |
|-----|---------|-------------|
|`model_config_folder`|`"model_configs"`|Folder where the configurations of each study are saved|
|`models_to_train`|`[]`|List of filenames within `model_config_folder`. If non-empty, the report will only be generated using the specified files.|
|`train_log_dir`|`"training_logs"`|Directory where after each training step, the summary of the training process will be stored as `.json`.|
|`documenttitle`|`{"title": "MNIST Training Example",<br>"author": "Test Document Author"}`|The title and the author of the final `.pdf` document.|



## Example
To provide a working example, this repository contains a minimal LaTeX template and different simple _keras_ models for MNIST. The full report can be found in `reports/report_MNIST.pdf` and `reports/report_MNIST.tex` for the rendered `.tex` file.
