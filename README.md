# Machine Learning - Group Project #1

## General Information

The repository contains the code for Machine Learning course 2022 (CS-433) project 1, the [Higgs Boson challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards)

> You can learn more by reading the [project description](ProjectDescription.pdf)

### Our Team

The 'MLE' team roster:
- Matteo Suez: [@matteosz](https://github.com/matteosz)
- Leonardo Trentini: [@leotrentini22](https://github.com/leotrentini22)
- Emanuele Nevali: [@emanuelenevali](https://github.com/emanuelenevali)

> Our team is called MLE, which is both an acronym of our names (Matteo, Leonardo, Emanuele)and a tribute to the [Maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) :grin:

### Data

The data `train.csv` and `test.csv` can be found in [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files). Before running the code, please download and place them in the `data` folder

In this folder can also be found `sample_submission.csv`, with all the results generated by the model and used for the final submission on AIcrowd


### Project Structure

- In `implementations.py` we implemented the 6 requested methods to train our model
- `model_helpers.py` contains useful functions for the implementation of methods
- `data_helpers.py` contains the functions implemented for feature engineering and preprocessing
- `HiggsChallenge.ipynb` include the complete notebook for our model, with cross validation and grid search for best parameters
- `run.py` uses the selected function to generate the results

### Report

The file `report.pdf` contains a complete and concise report of our work
