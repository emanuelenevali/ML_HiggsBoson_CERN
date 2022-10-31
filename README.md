<div id="top"></div>

<br />
<div align="center">
<h3 align="center">Higgs Boson Classification Project</h3>
  <p align="center">
    Machine Learning (CS-433)
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#General-Information">General Information</a></li>
    <li><a href="#Our-Team">Our Team</a></li>
    <li><a href="#Data">Data</a></li>
    <li><a href="#Structure">Structure</a></li>
    <li><a href="#Documentation">Documentation</a></li>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#Results">Results</a></li>
  </ol>
</details>

## General Information

The repository contains the code and report for the Machine Learning (CS-433) project #1, the [Higgs Boson challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards)

> You can learn more by reading the [project description](ProjectDescription.pdf)

## Our Team

The ***MLE*** team roster:
- Matteo Suez: [@matteosz](https://github.com/matteosz)
- Leonardo Trentini: [@leotrentini22](https://github.com/leotrentini22)
- Emanuele Nevali: [@emanuelenevali](https://github.com/emanuelenevali)

> Our team is called __MLE__, which is both an acronym of our names (Matteo, Leonardo, Emanuele) and a tribute to the [Maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) :grin:!

## Data

The data `train.csv`, `test.csv` and `sample-submission.csv` can be found in [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files). Before running the code, please create a folder named `data`, download the files and move them in the folder

- `train.csv` is the file containing the labeled data to train our model
- `test.csv` is the file containing unlabeled data, used to make our predictions
- `sample_submission.csv` is the file used to store our predictions and used for the final submission on AIcrowd

## Structure

- In `implementations.py` we implemented the 6 requested methods to train our model
- `model_helpers.py` contains useful functions for the implementation of methods
- `data_helpers.py` contains functions implemented for feature engineering and preprocessing of data
- `HiggsChallenge.ipynb` implements cross validation and grid search for the best hyperparameters and plots the charts used in the report
- `run.py` uses the selected function among the 6 presented in `implementations.py`, as well as the best hyperparameters found in `HiggsChallenge.ipynb` to generate our predictions

## Documentation

- The file `ProjectDescription.pdf` contains all the useful information for the project, such as the track and the guidelines
- The file `report.pdf` contains a complete and concise report of our work

## Usage

1. Clone the repo
   ```sh
   git clone https://github.com/emanuelenevali/ML_HiggsBoson_CERN
   ```
2. Follow the instructions contained in the [Data](#data) section
3. From this folder run in the terminal:
   ```sh
   python run.py
   ```
   or
   ```sh
   ./run.py
   ```

## Results

As a final result, we obtained an accuracy of **0.838** on the train set and our final submission scored **0.837** on the test set (with a F1 score of **0.751**), for a final position of 12 over 224 teams!

<p align="right">(<a href="#top">Back to top</a>)</p>