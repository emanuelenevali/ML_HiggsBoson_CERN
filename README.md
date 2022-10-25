# Machine Learning - Group Project #1

## General Information

The repository contains the code for Machine Learning course 2022 (CS-433) project 1, the [Higgs Boson challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards).

### Our Team

The 'MLE' team roster:
- Matteo Suez: [@matteosz](https://github.com/matteosz)
- Leonardo Trentini: [@leotrentini22](https://github.com/leotrentini22)
- Emanuele Nevali: [@emanuelenevali](https://github.com/emanuelenevali)

> Our team is called MLE, which is both an acronym of our names (Matteo, Leonardo, Emanuele)and a tribute to the [Maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) :grin:

### Data

The data `train.csv` and `test.csv` can be found in [ML_course/project1/data](https://github.com/epfml/ML_course/tree/master/projects/project1/data). Before running the code, please download and place them in the `data` folder


### Project Structure

- Firstly, we present our [implementation](implementation/implementation.py) of some methods extremely useful in the ML fields:

    1. `mean squared error gd(y, tx, initial w, max iters, gamma)`
    2. `mean squared error sgd(y, tx, initial w, max iters, gamma)`
    3. `least squares(y, tx)`
    4. `ridge regression(y, tx, lambda )`
    5. `logistic regression(y, tx, initial w, max iters, gamma)`
    6. `reg logistic regression(y, tx, lambda, initial w, max iters, gamma)`

- Then we faced a ML competition hosted by the [AIcrowd](https://www.aicrowd.com/) platform, based on the [dataset](http://opendata.cern.ch/record/328) from CERN on the Higgs Boson.

> You can learn more by reading the [project description](ProjectDescription.pdf).
