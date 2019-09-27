**keras_reports**: Automatic generation of training reports and model comparison in keras
======

<!-- TOC -->

- [Why](#why)
- [Example](#example)
    - [Example screenshots](#example-screenshots)
- [Getting started](#getting-started)
- [Configuration](#configuration)
    - [Configuration of the reporting tool `report_config.json`](#configuration-of-the-reporting-tool-report_configjson)
    - [Configuration of each model `model_configs/X.json`](#configuration-of-each-model-model_configsxjson)
        - [`data` field](#data-field)
        - [`model` field](#model-field)
        - [`training` field](#training-field)
        - [`report` field](#report-field)
- [Important modules](#important-modules)
- [How to implement a new model](#how-to-implement-a-new-model)

<!-- /TOC -->

## Why
When training Neural networks, we compare differnet choices of hyperparameters and architectures to determine which model performs best. To compare the different results, metrics need to be written down and put into a table by hand for presentation.

This package implements a practical all-in-one solution for _keras_ sequential models. It trains the models with their respective configuration and writes a `.pdf` directly, which contains all information, history plots, a table comparing metrics of all the different models at once, a model summary for all used architectures, etc.

Training accuracy, loss, and learning rate plots are written in `tikz` and passed to LaTeX template.

The `.pdf` is rendered from a LaTeX template using Jinja2. This means that the template is __fully customisable__ to fit into conference `.tex` templates, or corporate report layouts.

This repository only contains a __dummy MNIST example__. The generalisation should be straight-forward from this manual, contact me otherwise.

## Requirements
Apart from Python 3, you need a working `LaTeX`+`pdflatex` installation, e.g. [TeXLive](https://www.tug.org/texlive/)

## Example
To provide a working example, this repository contains a minimal LaTeX template and different simple _keras_ models for MNIST. The full report can be found in `reports/report_MNIST.pdf` and `reports/report_MNIST.tex` for the rendered `.tex` file. Hyper-references are included.

### Example screenshots
This is how the frontpage looks like using a blank LaTeX template:
![Frontpage](/example_screenshots/frontpage_example.png "Report frontpage")

This is how the training plots like using a blank LaTeX template (not the dummy MNIST example implemented in this repo, but one of the models I am testing for other purposes):
![Training plots](/example_screenshots/plots_example.png "Training history plots")

This is how the study details get listed using a blank LaTeX template:
![Study details](/example_screenshots/details_example.png "Study details")

This is how the model architecture gets listed using a blank LaTeX template:
![Architecture](/example_screenshots/arch_example.png "Model architecture")

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
|`model_config_folder`|`"model_configs"`|Folder where the configurations of each study are saved.|
|`models_to_train`|`[]`|List of filenames within `model_config_folder`. If non-empty, the report will only be generated using the specified files.|
|`train_log_dir`|`"training_logs"`|Directory where the report generator will search for `.json` files.|
|`documenttitle`|`{"title": "YOURTITLE",`<br>`"author": "YOURNAME"}`|Title and author of the final `.pdf` document.|
|`render_folder`|`"reports"`|Directory, where the rendered `.tex` and `.pdf` will be stored.|
|`report_filename`|`"report"`|Filename (without ending) of the rendered `.tex` and `.pdf`.|

### Configuration of each model `model_configs/X.json`
These jsons have four subsections `data`, `model`, `training`, `report`. For your personal needs, you can add fields as you wish.

#### `data` field
| Key | Type | Default | Description |
|-----|---------|---------|-------------|
|`name`|`str`|no default|Name of your dataset (will appear in report).|
|`image_size`|list of `int`, length=2|If you work with image data like I do, here is where you specify to what size the inputs will be downsampled.|

#### `model` field
Depending on how you are parametrising your model and the layers you are using, different keys will arise here. Attached is an example for MLPs and a simple ConvNet.

| Key | Type | Description |
|-----|---------|-------------|
|`name`|`str`|Name of your model (will appear in report).|
|`filter_sizes`|list of `int`|For each convolutional layer, a filter size|
|`n_hidden_layers`|`int`|In MLPs where you only work with fully connected layers, this is the number of layers used.|
|`dense_units`|list of `int`|Per fully connected layer, a number of nodes inside this layer.|
|`dropout_rates`|`float` between 0 and 1|Per dropout layer, a `keep_prob` between 0 and 1.|
|`n_classes`|`int`|Number of output classes or masks (used as parameter in model definition).|
|`is_saved`|`boolean`|If `true`, the weights will be saved after training.|
|`model_folder`|`str`|If `is_saved` is `true`, this specifies the directory where the weights will be saved.|

#### `training` field
Depending on how you are parametrising your model and the layers you are using, different keys will arise here. Attached is an example for MLPs and a simple ConvNet.

| Key | Type | Description |
|-----|---------|-------------|
|`optimizer`|`str`|Name of the optimizer you want to use. If you use a _keras_ implemented one, the `.pdf` report will contain author and year of the related publication.|
|`optim_config`|`dict`|Configuration of the optimizer specified above. All unspecified values will lead to default values being used. The `.pdf` report will list all parameters and the configuration used in training.|
|`loss`|`str`|Name of the loss function to be used. Custom loss functions can be defined, but must be imported to `nn_wrapper.py`.|
|`metrics`|list of `str`|The metric _keras_ will use in training ([RTFM](https://keras.io/metrics/)).|
|`n_epochs`|`int`|Number of epochs.|
|`batch_size`|`int`|Batch size ([RTFM](https://keras.io/models/sequential/#fit)).|
|`shuffle`|`boolean`|Batches will be reshuffled if set to `true` ([RTFM](https://keras.io/models/sequential/#fit)).|

#### `report` field
This contains some info about the reporting tool that will be displayed but is by itself unrelated to the models or the optimisation.

| Key | Type | Description |
|-----|---------|-------------|
|`study_name`|`str`|Name of the study configured in this file. This name will be printed to the `.pdf`.|
|`link`|`str`|If this string is non-empty (length greater than 0), it will be printed on the `.pdf` as a link.|
|`export_folder`|`str`|This is where the training summary `.json` will be saved. Should coincide with `train_log_dir` in the report configuration file. The possibility to assign different folders to these keys allows for a modular code structure and independent execution of training and report generation.|
|`datetimeformat`|`str`|Datetime format used in the training log `.json` ([RTFM](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)).|
|`datetimeformat_report`|`int`|The datetime format which will be used in the report ([RTFM](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)).|


## Important modules
1. __`train.py`__ Contains a function `main(report_config='./report_config.json', render_reports=True)`:
   * `report_config` Str, points to the report configuration file
   * `render_reports`: Boolean, if True, it will already run the report generation after training is over (concretely, it will run the funciton `make_reports` in `report.py`).
2. __`report.py`__ Contains a function `main(report_config='report_config.json')`:
   * `report_config` Str, points to the report configuration file.
3. __`dataset/dataset.py`__ Implement the dataset as class readable by `nn_wrapper.py`.
4. __`nn_wrapper.py`:__ Contains a wrapper class `NeuralNetWrapper(dataset, model, config)`. Reads configuration file, model and the dataset. Contains a method `.train()` which will train the model according to the specified configuration, and a method `.to_json` which will export the training log to the directory specified in the configuration.

## How to implement a new model
1. Add your model to `models/models.py`. I usually define custom layers in a `layers.py` in the same subfolder, but that is up to you.
2. _In case you use layers which are not yet implemented,_ choose what configuration will be printed in the report by adding a configuration to the endless `if-elif-else` section in `nn_wrapper` (in the definition of the method `.to_json()` to the class `NeuralNetWrapper`).
3. _In case you use layers which are not yet implemented,_ define a colour in which the layer will be displayed. Add it to the colour definition section in the beginning of the template section of `latex/template.tex`.
4. Add your model to the function `choose_model()` in `train.py`.

Layers which are already implemented ([RTFM](https://keras.io/layers/core/)): `InputLayer`, `Conv2D`, `BatchNormalization`, `Dense`, `UpSampling2D`, `MaxPooling2D`, `Activation`, `Dropout`, `Concatenate`, `Flatten`, `Add`.
