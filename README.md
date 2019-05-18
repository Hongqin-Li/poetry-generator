# Assignment 3 Report

## Part 1

In this part, I just stick into [source code](https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py) of Pytorch, which implements some variances of RNN such as LSTM.

## LSTM

$$
\begin{array}{ll} 
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}
$$

where $h_t$ is the hidden state at time $t$, $c_t$ is the cell state at time $t$, $x_t$ is the input at time $t$, $h_{(t-1)}$ is the hidden state of the layer at time $t-1$ or the initial hidden state at time $0$, and $i_t$, $f_t$, $g_t$, $o_t$ are the input, forget, cell, and output gates, respectively. $\sigma$ is the sigmoid function, and $*​$ is the *Hadamard product*.

> Hadamard product is a element-wise product, such that $(AB)_{ij} = a_{ij}b_{ij}$



## Part 2

