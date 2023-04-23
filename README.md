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
| Glorot   | None, tanh, sigmoid, softmax             | $1/fan_{avg}$ |
| He       | ReLU, Leaky ReLU, ELU, GELU, Swish, Mish | $2/fan_{in}$  |
| LeCun    | SELU                                     | $1/fan_{in}$  |

## Clustering Algorithms applications

### Dimensionality reduction
Once a dataset has been clustered, it is usually possible to measure each instance's affinity with each cluster. Each instances's feature vector **x** can then be replaces with the vector of its cluster affinities. If there are $k$ clusters, then this vector is k-dimensional. The new vector is typically much lower-dimensional but it can preserve enough information.

### Feature engineering
The cluster affinites can often be useful as extra features.

## Losses, Optimizers, Activation Functions
We use the "sparse_categorical_crossentropy" loss because we have sparse labels (i.e., for each instance, there is just a target class index, from 0 to 9), and classes are exclusive. If instead we had one target probability per class for each instance (such as one-hot vectors, e.g., [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] to represent class 3), then we would need to use the "categorical_crossentropy" loss instead.

If we were doing binary classification or multilable binary classification, then we would use the "sigmoid" activation function in the output layer instead of "softmax" activation function, and we would use the "binary_crossentropy" loss.

## Running TF Serving in a Docker Container
``sh
docker pull tensorflow/serving
``

``sh
docker run -it --rm -v "/home/raha/Desktop/DNN/my_mnist_model:/models/my_mnist_model" -p 8500:8500 -p 8501:8501 -e MODEL_NAME=my_mnist_model tensorflow/serving
``

1. -it: Makes the container interactive and displays the server's output
2. --rm
3. -v: Makes the host's my_mnist_model directory available to the container at the path /models/my_mnist_model
4. -p: The Docker image is configured to use port 8500 to serve the gRPC API and 8501 to serve the REST API by default.
5. -e: Sets the container's MODEL_NAME environment variable, so TF Serving knows which model to serve. By default, it will look for models in the /models directory and it will automatically serve the latest version it finds.

### Json vs gRPC
| Aspect   | Json     | gRPC     | extra explanation |
| -------- | -------- | -------- |
| Glorot   | None, tanh, sigmoid, softmax             | $1/fan_{avg}$ |
| He       | ReLU, Leaky ReLU, ELU, GELU, Swish, Mish | $2/fan_{in}$  |
| LeCun    | SELU                                     | $1/fan_{in}$  |

## Learning rate Scheduling
If you set it slightly too high, it will make progress very quickly at first, but it will end up dancing around the optimum and never really settling down


### Power Scheduling
A function of the iteration number, $t: \eta(t) = \eta_0 / (1 + t/s)^c$

### Exponential Scheduling
$\eta(t) = \eta_0 0.1^{t/s}$

### Piecewise Constant Scheduling
Use a constant learning rate for a number of epochs then a smaller learning rate for another number of epochs

### Performance Scheduling
Measure the validation error every N steps and reduce the learning rate by a factor of $\lambda$ when the error stops dropping

## Output Layer Activation Function
1. An MLP may not have any activation function for the output layer, so it's free to output any value, this is generally fine.
2. If you want to guarantee that the output will always be positive, then you should use the ReLU activation function in the output layer, or the softplus activation function, which is a smooth variant of ReLU.
3. If you want to guarantee that the predictions will always fall within a given range of values, then you should use the sigmoid function or the hyperbolic tagent and scale the targets to the appropriate range