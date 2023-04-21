# DNN
## Glorot and He Initialization
They proposed a way to alliviate the unstable gradients problem. We need the signal to flow properly in both direction (forward and backward). We dont' want the signal to die out, nor do we want it to explode and saturate. For the signal to flow properly, we need the variance of the outputs of each layer to be equal to the variance of its inputs.(Imagine a chain of amplifiers, your voice has to come out of each amplifier at the same amplitude as it came in) and we need the gradients to have equal variance before and after flowing through a layer in the reverse direction. It is not possible unless fan-in = fan-out, but Glorot and Bengio proposed a good compromise that has proven to work very well in practice: the connection weights of each layer must be initialized randomly as described below, where: $$fan_{avg} = (fan_{in} + fan_{out})/2$$

Normal distribution with mean 0 and variance: $$\sigma^2 = 1/fan_{avg}$$
Or a uniform distribution between $-r$ and $+r$, with $$r = \sqrt{3/fan_{avg}}$$

## Pseudoinverse
Using SVD, we decomposition matrix $X$ into $U \sum V^T$. 
$$X^+ = V \sum^+ U^T$$
To compute $\sum^+$ the algorithm takes $\sum$ and sets to zero all values smaller than a tiny threshold value, then it replaces all nonzero values with their inverse and finally it transposes the resulting matrix

## Initialization
| Initialization | Activation functions | $\sigma^2$ (Normal) |
| -------- | -------- | -------- |
| Glorot | None, tanh, sigmoid, softmax             | $1/fan_{avg}$ |
| He     | ReLU, Leaky ReLU, ELU, GELU, Swish, Mish | $2/fan_{in}$  |
| LeCun  | SELU                                     | $1/fan_{in}$  |

## Clustering Algorithms applications

### Dimensionality reduction
Once a dataset has been clustered, it is usually possible to measure each instance's affinity with each cluster. Each instances's feature vector **x** can then be replaces with the vector of its cluster affinities. If there are $k$ clusters, then this vector is k-dimensional. The new vector is typically much lower-dimensional but it can preserve enough information.

### Feature engineering
The cluster affinites can often be useful as extra features.

## Losses, Optimizers, Activation Functions
We use the "sparse_categorical_crossentropy" loss because we have sparse labels (i.e., for each instance, there is just a target class index, from 0 to 9), and classes are exclusive. If instead we had one target probability per class for each instance (such as one-hot vectors, e.g., [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] to represent class 3), then we would need to use the "categorical_crossentropy" loss instead.

If we were doing binary classification or multilable binary classification, then we would use the "sigmoid" activation function in the output layer instead of "softmax" activation function, and we would use the "binary_crossentropy" loss.
