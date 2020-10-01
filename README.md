# Pneumonia-Detection

Medical imaging refers to a set of techniques that aim
to produce visual representations of internal aspects of the
human body in a noninvasive manner. Medical imaging seeks
to reveal the internal organs masked by the skin and bones so
that they can be analysed for any abnormalities and diagnose
and treat disease. The purpose of
this project was to develop an automated pneumonia
diagnosis system using supervised learning that indicates whether a given chest X-ray
image represents a patient that has pneumonia or not. A detailed write up can be found in the report.

![Alt text](https://github.com/phantom820/Pneumonia-Detection/blob/master/results/chest_x_ray.png)

### Supervised learning paradigms used
- k Nearest Neighbours
- Logistic regression (gradient descent learning and genetic algorithm learning)
- Multilayer perceptron (gradient descent learning)

### Model perfomance
The multilayer perceptron is the best perfoming model with a peak accuracy
of 86.31% followed by the logistic regression model. The confusion matrices are shown below

#### Multilayer perceptron
![Alt text](https://github.com/phantom820/Pneumonia-Detection/blob/master/results/mp-matrix.png)

#### Logistic regression (gradient descent learning)
![Alt text](https://github.com/phantom820/Pneumonia-Detection/blob/master/results/lr-c.png)
![Alt text](https://github.com/phantom820/Pneumonia-Detection/blob/master/results/lr-matrix.png)

#### Logistic regression (genetic algorithm learning)
![Alt text](https://github.com/phantom820/Pneumonia-Detection/blob/master/results/lr-g-cost.png)
![Alt text](https://github.com/phantom820/Pneumonia-Detection/blob/master/results/lr-g-matrix.png)
