# Phase 2
In this phase, we wanted to reach higher speedups using cuda. I did not really get the results I wanted and didn't gain any speedup. What I did for this phase was to implement feedforward using gpu but it is extremly slow and we cannot use it. After I gave up on backpropagation, I went on to the hill climbing method, but, did not get any good results either.

## brief explanation
As I mentioned, there are two sides to this phase:
1) cuda implementation of feedforward
2) hill climbing algorithm

To implement feedforward in cuda, I only needed a kernel for matrix multiplication which I have implemented in `genann_helper.cu`. Then in function `cudaFeedForward`, I use this kernel to compute each layer's output until the results are computed.

To implement hill climbing algorithm, we first initialize a neural network, then we add some random noise to the weights, if we have a worse network, we bring back the old weights. Otherwise, we use those new weights from now on. We do this for a specific number of times or until our sum of squared error is lower than some threshold. This whole algorithm is implemented in a function called `genann_hillclimb` in `genann_c.c` which is the modified version of the genann library.
On the cuda's side, we only need to add gaussian noise to every weight of the neural network each time. To do that, I implemented a kernel in `genann_helper.cu` and then, I use that kernel in `cudaRandomWeights` function and then, use that function in `genann_hillclimb`.