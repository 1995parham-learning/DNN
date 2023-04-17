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
| Initialization | Activation functions | \sigma^2 (Normal) |
| -------- | -------- | -------- |
| Row 1, Column 1 | Row 1, Column 2 | Row 1, Column 3 |
| Row 2, Column 1 | Row 2, Column 2 | Row 2, Column 3 |
| Row 3, Column 1 | Row 3, Column 2 | Row 3, Column 3 |
