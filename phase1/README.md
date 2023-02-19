# Phase 1
In this phase we tried to add parallel computing with OpenMP to the genann library.

## brief explanation
There are two main sections we can add parallelism to: 
1) genann_run
2) genann_train

In `genann_run` feed forward happens. When we are doing feedforward, each layers outputs depend on outputs of the previous layer, so we should only use multiple threads to compute outputs in a single layer and after all of them are finished, go on to the next layer.

In `genann_train` gnann implements backpropagation. In this scenario, we have to compute deltas and then we compute gradients based on those deltas. We cannot compute deltas and gradients in parallel so we first have to compute the deltas then the gradients. We can compute deltas for every layer in parallel but we have to consider the dependency every layer has on the next layer. To deal with this, we start from the last layer, compute its deltas in parallel, then we go for the previous layer and continue that way. After all deltas are computed, we use parallelism to update weights in the same way we did for deltas. We update last layer's weights in parallel, then we go for the layer before and so on.