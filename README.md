# multicore-processing-fcnn
In this project, we wanted to use our knowledge of parallel programming to improve genann library's speed. 

## Genann
Genann is a minimal implementation of a neural network initialization, feedforward and backpropagation and all its other helper functions. With the help of this library you can initialize and train a neural network.
Here's a link to [Genann's repo](https://github.com/codeplea/genann).

## Phases
This project had two phases.

In phase 1, we had to use OpenMP library to implement genann methods in a parallel way. You can find more about this phase in [here](./phase1/).

In phase 2, we tried to gain more speedup using gpu and cuda programming. After spending time on trying to implement backpropagation in gpu, because of how much data dependancy exists in this step, I gave up on backpropagation and tried to use a simple hill climbing method to train my network. More details on this phase can be found [here](./phase2/).

## Data
We used Semeion Handwriting Digit dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php). Read more about this dataset in [here](./data/).