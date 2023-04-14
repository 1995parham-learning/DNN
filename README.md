# DNN
## Glorot and He Initialization
They proposed a way to alliviate the unstable gradients problem. We need the signal to flow properly in both direction (forward and backward). We dont' want the signal to die out, nor do we want it to explode and saturate. For the signal to flow properly, we need the variance of the outputs of each layer to be equal to the variance of its inputs.(Imagine a chain of amplifiers, your voice has to come out of each amplifier at the same amplitude as it came in) and we need the gradients to have equal variance before and after flowing through a layer in the reverse direction. It is not possible unless fan-in = fan-out, but Glorot and Bengio proposed a good compromise that has proven to work very well in practice: the connection weights of each layer must be initialized randomly as described in Equation 1, where $fan_{avg} = (fan_{in} + fan{out})/2$.

<center>
  Normal distribution with mean 0 and variance 
  $$
  \sigma^2 = 1/fan_{avg}
  $$
</center>

