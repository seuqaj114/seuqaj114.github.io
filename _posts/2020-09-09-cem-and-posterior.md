---
title:  "The cross-entropy method for posterior estimation"
mathjax: true
layout: post
---


The __cross-entropy method (CEM)__ is a simple and elegant algorithm used to solve many types of optimization problems such as combinatorial optimization, traveling salesman, and quadratic assignment problems. It is particularly useful for rare-event simulation/estimation, where we want to accurately estimate very small probabilities. 

The CEM method is frequently used in the machine learning literature as a tool to perform model-predictive control. However, I recently found that it can also be very useful for __posterior estimation__, particularly when the posterior is intractable. In this post, I will explain how to use and derive it from a probabilistic perspective[^1]. Hopefully, you will get a feel for how simple and general it is, so you can start using it in your own models.


----------------

## Motivation

During my research in physical parameter estimation from video, I ran into a very basic problem: given the trajectory of a bouncing ball, what are the physical parameters and initial conditions that generate the observed trajectory? The physical parameters here can be the y-position of the ground-level, $h$, and the restitution coefficient of the ball, $\epsilon$ (how much it bounces), and the initial parameters are the initial position and velocity. Since we can simulate this type of trajectory using a simple physics engine, we can use some optimization/search algorithm to find the parameters that minimize the difference between the observed and simulated trajectories. Let's formalize this.

Let $x_{1:T}^{\text{obs}}$ be a noisy positions sequence of length $T$. Given a simulator, $x_{1:T}^{\text{sim}} = f(\theta)$, parametrized a vector of physical parameters and initial conditions, $\theta$, we want to maximize the likelihood of the observed trajectory under the simulator:

$$ \theta^* = \text{argmax}_{\theta} \; p(x^{\text{obs}}_{1:T}| f(\theta)) $$ 

In the contexts of physics or robot modeling, this problem is usually called _(non-linear) system identification_, so google this term if you want to find more about this topic.

For ease of modelling we will assume that the simulation is deterministic, hence the points are conditionally independent given the parameters. This allows us to write the likelihood of a sequence simply as the product of the likelihoods of each step:

$$ p(x^{\text{obs}}|\theta) = \prod_{t=1}^T p(x^{\text{obs}}_t| x^{\text{sim}}_t(\theta)) = \prod_{t=1}^T \mathcal{N}(x^{\text{obs}}_t|x^{\text{sim}}_t(\theta), \sigma^2) $$

where we assumed that for each time-step, the likelihood is a Gaussian centered at the simulated position. If the simulator was stochastic, estimating the sequence likelihood $p(x^{\text{obs}}\|\theta)$ would become a significantly harder problem, which would require using Kalman or particle filters to compute the marginal likelihood under some latent variable model[^2]. We also assume that the simulator $f(\theta)$ is a black-box, so we can't exploit the structure of the equations of motion to aid inference and learning. This is unrealistic for such a simple case as a bouncing ball, but it is a common setting in [simulation-based inference](https://www.pnas.org/content/early/2020/05/28/1912789117).

In this post, I will show how we can use the cross-entropy method to estimate the posterior over the parameters, $p(\theta\| x^{\text{obs}}_{1:T})$, and not just the maximum likelihood parameters $\theta^*$. Estimating the posterior is useful if we want to perform Bayesian inference or measure the uncertainty of the parameters.



---------------------

## Method


As before, let's consider a very simple ball falling vertically and bouncing off the floor. This system's state is described by a 1D position $x$ and velocity $v$. The physical parameters are the ground height $h$ and restitution coefficient $\epsilon \in [0, 1]$. The ball falls with a gravity value $g=9.8$.

The simulator is defined as a simple 2nd order Euler integrator of a falling ball with inelastic bounce at $h$:
~~~ python
def bouncing_ball_step(x, v, g, h, eps, dt, iters=50):
    for _ in range(iters):
        bounce = x <= h
        if np.any(bounce):
            v[bounce] = -eps[bounce] * v[bounce]
            x[bounce] = h[bounce] + (h[bounce] - x[bounce])
        v = v - dt * g / iters
        x = x + dt * v / iters
    return x, v
~~~

If we generate a sequence with added Gaussian noise, true parameters $\theta^* = [h, \epsilon] = [0.0, 0.8]$ and initial conditions $[x_0, v_0]=[12.0, 0.0]$, we get the following trajectory $(T=150$, $dt=0.1)$: 
{:refdef: style="text-align: center;"}
![e2c-trajectory](/assets/figures/cem/trajectory.png){:width="50%"}
{: refdef}


Since we want to estimate $p(\theta\|x^{\text{obs}})$ we start by visualizing the likelihood function $p(x^{\text{obs}}\|\theta)$. Plotting the likelihood for a 2D grid of values $[h, \epsilon] \in [-3, 3] \times [0.2, 1]$, we get the following heatmap:

{:refdef: style="text-align: center;"}
![e2c-trajectory](/assets/figures/cem/posterior.png){:width="50%"}
{: refdef}

Since we can see that the likelihood function is approximately Gaussian near the maximum, approximating the posterior with a Gaussian distribution is a reasonable choice.
Therefore, we define a distribution $q(\theta|\mu,\Sigma) = \mathcal{N}(\theta|\mu, \Sigma)$ parametrized by $\mu$ and $\Sigma$, which we will use to approximate the true posterior $p(\theta\|x_{1:T})$. 

The simplest way to perform this approximation is by minimizing the forward KL-divergence between the true and the approximate posterior w.r.t $\mu$ and $\Sigma$:

$$ \hat{\mu}, \hat{\Sigma} = \text{argmin}_{\mu, \Sigma} \text{KL}(p(\theta|x_{1:T}) \| q(\theta|\mu,\Sigma)) = \text{argmin}_{\mu, \Sigma} \int_{\theta} p(\theta|x_{1:T}) \log \frac{p(\theta|x_{1:T})}{q(\theta|\mu,\Sigma)} \text{d}\theta $$

Removing terms that don't depend on $\mu$ and $\Sigma$ we get:

$$ \text{argmin}_{\mu, \Sigma} \int_{\theta} \textcolor{blue}{p(\theta|x_{1:T})} \log \frac{p(\theta|x_{1:T})}{\textcolor{green}{q(\theta|\mu,\Sigma)}} \text{d}\theta = 
\text{argmin}_{\mu, \Sigma} \int_{\theta} \textcolor{blue}{p(\theta|x_{1:T})} \log \textcolor{green}{q(\theta|\mu,\Sigma)} \text{d}\theta \quad \text{(1)} $$


From Bayes' rule we known that the posterior can be written as:

$$ p(\theta|x_{1:T}) = \frac{p(x_{1:T}|\theta)p(\theta)}{p(x_{1:T})} \quad , \quad p(x_{1:T}) = \int p(x_{1:T}|\theta)p(\theta)\text{d}\theta. $$

However, the denominator $p(x_{1:T})$ is typically intractable, so we can only know the posterior up to a constant, $p(\theta\|x_{1:T}) \propto p(x_{1:T}\|\theta)p(\theta)$. Fortunately, since $p(x_{1:T})$ does not depend on $\theta$ or $\mu$ or $\Sigma$, we can simplify (1) to:

$$ \text{argmin}_{\mu, \Sigma} \int p(\theta|x_{1:T}) \log q(\theta|\mu,\Sigma) \text{d}\theta = 
\text{argmin}_{\mu, \Sigma} \int p(x_{1:T}|\theta)p(\theta) \log q(\theta|\mu,\Sigma) \text{d}\theta $$

This gives us an optimization objective that depends only on quantities we can evaluate (the likelihood, the prior, and the approximate posterior).
Since $q$ is Gaussian, we can easily take the derivatives of the integral with respect to $\mu$ and $\Sigma$ and set them to zero, resulting in:

$$
\begin{aligned}
\hat{\mu} &= \frac{\int \theta \, p(x_{1:T}|\theta)p(\theta) \text{d}\theta} {\int p(x_{1:T}|\theta)p(\theta) \text{d}\theta} \\
\hat{\Sigma} &= \frac{\int (\theta^{(n)}-\hat{\mu})(\theta^{(n)}-\hat{\mu})^T \, p(x_{1:T}|\theta)p(\theta) \text{d}\theta} {\int p(x_{1:T}|\theta)p(\theta) \text{d}\theta}
\end{aligned}
$$

There is only one problem: we still need to evaluate these integrals via Monte-Carlo integration. However, using samples from the prior $p(\theta)$ to estimate these quantities would be highly innefficent, since the prior would have to be close to the region of high probability mass of the posterior. Instead, we use the approximate posterior itself to estimate the integrals via [importance sampling](https://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/slides.pdf) (slide 14):

$$ \int g(\theta) \, \textcolor{blue}{p(x_{1:T}|\theta)p(\theta)}  \text{d}\theta = \int g(\theta) \, q(\theta|\mu,\Sigma) \frac{\textcolor{blue}{p(x_{1:T}|\theta)p(\theta)}}{q(\theta|\mu,\Sigma)} \text{d}\theta \approx \sum_{n=1}^N g(\theta^{(n)}) \, \frac{p(x_{1:T}|\theta^{(n)})p(\theta^{(n)})}{q(\theta^{(n)}|\mu,\Sigma)} $$

with $\theta^{(n)} \sim q(\theta\|\mu,\Sigma)$.

Substituting this approximation in the estimates for $\hat{\mu}$ and $\hat{\Sigma}$ above we get (after simplification):

$$
\begin{aligned}
\hat{\mu} &= \sum_n \textcolor{blue}{\bar{w}_n} \theta^{(n)} \\
\hat{\Sigma} &= \sum \textcolor{blue}{\bar{w}_n} (\theta^{(n)}-\mu)(\theta^{(n)}-\mu)^T
\end{aligned}
$$

where $\textcolor{blue}{\bar{w}_n} = \frac{\textcolor{green}{w_n}}{\sum_n \textcolor{green}{w_n}}$ and $\textcolor{green}{w_n} = \frac{p(\mathbf{x}\|\theta^{(n)})p(\theta^{(n)})}{q(\theta^{(n)}\|\mu,\Sigma)}$.


By performing this estimation repeatedly, we get an iterative procedure for estimating $\mu$ and $\sigma$ via adaptive importance sampling. Though there were a bit of maths above, the pseudo-code below shows that this algorithm is extremely easy to implement, as it involves only the computation of a weighted mean and covariance matrix.


------------------------

__CEM pseudocode__

1. Init $\hat{\mu} = \mu_{\text{prior}}$, $\hat{\Sigma} = \Sigma_{\text{prior}}$;
2. Repeat until convergence:
    * Draw $N$ samples $\theta^{(n)} \sim q(\theta\|\hat{\mu},\hat{\Sigma})$;
    * For each sample $\theta^{(n)}$, evaluate the likelihood $p(\mathbf{x}\|\theta^{(n)})$, the prior $p(\theta^{(n)})$ and the posterior $q(\theta^{(n)})$;
    * Reestimate $\mu$ and $\Sigma$ as:
        * $\hat{\mu} = \sum_n \textcolor{blue}{\bar{w}_n} \theta^{(n)}$ 
        * $\hat{\Sigma} = \sum \textcolor{blue}{\bar{w}_n} (\theta^{(n)}-\mu)(\theta^{(n)}-\mu)^T$
        * where $\textcolor{blue}{\bar{w}_n} = \frac{\textcolor{green}{w_n}}{\sum_n \textcolor{green}{w_n}}$ and $\textcolor{green}{w_n} = \frac{p(\mathbf{x}\|\theta^{(n)})p(\theta^{(n)})}{q(\theta^{(n)}\|\hat{\mu},\hat{\Sigma})}$.
3. Return $\mathcal{N}(\theta\|\hat{\mu},\hat{\Sigma})$.

-----------------------

A very neat property of this algorithm is that it doesn't involve tuning any learning rate like in the usual gradient descent methods! All we need is to define the initial distribution (which we use as the prior), the number of samples per iteration, and a convergence criterion. Applying this algorithm to estimate $[h, \epsilon]$ in the bouncing ball setting above we get the following evolution of samples $\theta^{(n)}$ $(N=200)$:

{:refdef: style="text-align: center;"}
![e2c-trajectory](/assets/figures/cem/cem_iters.gif){:width="40%"}
{: refdef}

We can see that the samples move towards the regions of high likelihood as the iterations progress. After 30 iterations we get the estimates:

$$ \hat{\mu} = [-0.053, 0.794]  \quad ,\quad \hat{\Sigma} = 
\begin{bmatrix}
   1.568 & -0.053 \\
   -0.053 & 0.004
\end{bmatrix}
$$

which are quite close to the true parameters $[h, \epsilon] = [0.0, 0.8]$ :D.

The importance sampling estimate improves with an increased number of samples $N$, and when likelihood distribution is not too peaked around $\theta^*$. An excessively peaked likelihood will cause all the samples $\theta^{(n)}$ to have likelihood close to 0, resulting in a poor posterior approximation. For the same reason, it is important to choose a prior distribution (=initial estimate of the posterior) that has enough overlap with the true posterior. 


<!--
Having determined $q(\theta\|\mu,\Sigma)$, we can estimate the marginal likelihood of a trajectory under the model M as:

$$ 
\begin{aligned}
p(\mathbf{x}|M) &= \int p(\mathbf{x}|\theta, M)p(\theta|M) \; d\theta \\
&= \int q(\theta|\mu,\Sigma) \frac{p(\mathbf{x}|\theta, M)p(\theta|M)}{q(\theta|\mu,\Sigma)} \; d\theta \\
&\approx \frac{1}{N} \sum_n \frac{p(\mathbf{x}|\theta^{(n)}, M)p(\theta^{(n)}|M)}{q(\theta^{(n)}|\mu,\Sigma)}, \quad \theta^{(n)} \sim q(\theta\|\mu,\Sigma).
\end{aligned}
$$


For a set of motion models $\{M_1, M_2,..., M_J\}$, the probability that a trajectory corresponds to a motion model can be computed in the standard model comparison way:

$$ p(M=j|\mathbf{x}) = \frac{p(\mathbf{x}|M=j)p(M=j)}{\sum_{j'=1}^J p(\mathbf{x}|M=j')p(M=j')} $$
-->

----------------------

## Limitations, practical issues, and takeaways

* __Variance over-estimation__ &nbsp; Since we are minimizing the forward KL-divergence, the approximate posterior will try to cover all the probability mass of the true posterior. This leads to an over-estimation of the variance, especially if the posterior is multi-modal. Due to the properties of the KL divergence, minimizing the reverse KL-divergence instead, $\text{KL}(q(\theta\|\mu,\Sigma)\\|p(\theta\|x_{1:T}))$, as done in variational inference, would yield an approximate posterior that fits a single mode of the true posterior mode more accurately, but doesn't cover all the probability mass. You can find a better discussion on the trade-offs of each version in [this blog post](https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/).

* __Computational efficiency__ &nbsp; This procedure is very wasteful in terms of computational resources because we are discarding all the previous iterations' samples and computation. Instead of discarding all the samples, we could keep a list of the previously found good samples (in terms of likelihood), and progressively decrease the number of new samples per iteration. Further heuristics can be used to improve the efficiency of CEM, as proposed in [this paper](https://arxiv.org/abs/2008.06389), in the context of planning for control. We also assume that each simulation is relatively fast, which isn't always the case when modeling a complex physical system.

* __High dimensions__ &nbsp; Importance sampling (like other Monte-Carlo methods) become increasingly inefficient as the dimensionality of the space being sampled increases. This is because as the dimensions increase, it becomes less and less likely for any sample to be in a region of high likelihood. This is an important practical point to keep in mind since sometimes the sampler fails not because of some implementation error, but just because the space being sampled is just too high-dimensional or the likelihood too peaked.

The cross-entropy method has many more applications that I didn't discuss here, but hopefully, this will give you a good starting point from which to explore and understand this method. It is an incredibly versatile method that can be used anytime you want to estimate the distribution of parameters of some observed physical system where we have access to a model of the system. We assumed that we can compute the likelihood function, but when this is not possible it is necessary to resort to likelihood-free methods (as briefly described in [these slides](https://michaelgutmann.github.io/assets/slides/Gutmann-2017-04-21.pdf)). 


---------------------

Many thanks to André Melo ([Twitter](https://twitter.com/_andrenmelo)) for proofreading and suggestions.

---------------------

[^1]: Good tutorials on CEM for control and rare-event estimation include [this pdf](http://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf) (the definitive CEM guide, written by the original authors of CEM) and [this blog post](https://towardsdatascience.com/cross-entropy-method-for-reinforcement-learning-2b6de2a4f3a0).

[^2]: This problem is addressed in [this paper](https://www.sciencedirect.com/science/article/pii/S0005109810004279), but it is out of the scope of this post.

