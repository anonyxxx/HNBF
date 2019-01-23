# Hierarchical Negative Binomial Factorization
>When being exposed to an item in a recommender system, a user may consume the item (as known as success exposure) or may neglect it (as known as failure exposure). Most of the prior works on matrix factorization merely consider the former and omit the latter. In addition, classical methods which assume each observation over a Poisson cannot be feasible to overdispersed data. In this paper, we propose a novel model, hierarchical negative binomial factorization (HNBF), which models dispersion by a hierarchical Bayesian structure rather than assigning a constant to the prior of dispersion directly, thus alleviating the effect of data overdispersion to help with performance gain for recommendation. Moreover, we factorize the dispersion of zero entries approximately into two low-rank matrices, thus limiting the computational cost of updating per epoch linear to the number of nonzero entries. In the experiment, we show that the proposed method outperforms the state-of-the-art methods in terms of precision and recall on implicit data in recommendation tasks.

## Required library
- <a href="https://www.mathworks.com/matlabcentral/fileexchange/23576-min-max-selection" target="_blank">Min/Max selection</a> 

## Usage
Add the included folders to path and run
```matlab
test_FastHNBF
```
