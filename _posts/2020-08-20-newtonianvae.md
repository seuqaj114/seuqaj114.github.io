---
title:  "NewtonianVAE"
subtitle: "Proportional Control and Goal Identification from Pixels via Physical Latent Spaces"
layout: post
---

This blog post will explain the ideas behind my most recent [paper](https://arxiv.org/abs/2006.01959), _NewtonianVAE: Proportional Control and Goal Identification from Pixels via Physical Latent Spaces_, and provide some ideas for future work that either me (or you!) could tackle. This work was done in collaboration with Michael Burke ([Twitter](https://twitter.com/mgb_infers), [Scholar](https://scholar.google.com/citations?user=Abz56f4AAAAJ&hl=en)) and Tim Hospedales ([Twitter](https://twitter.com/tmh31), [Scholar](https://scholar.google.com/citations?hl=en&user=nHhtvqkAAAAJ)).

--------------------

## PID control

Before diving into the paper itself, I will quickly explain the concept of PID control for those of you who are not familiar with this area.

PID stands for Proportional-Integral-Derivative, which is a mouthful... Let's ignore the I and D terms, and focus just on P, as it is the most important one. 
So what is a __proportional (or P-) controller__? Let's imagine a very simple system like a ball rolling on a horizontal table, to which we can apply a force along the horizontal plane (for example, by pushing it with a blow drier). This sistem is described by the 2D position of the ball $x$ and the action is the force applied along this plane, $F$. 

If the ball is initially at a position $x_0$ and we want to move it towards a goal position $x^{\text{goal}}$, a simple way to achieve this is by applying, at each time $t$, a force of the form $F_t = c \cdot (x^{\text{goal}} - x_t)$, starting at $t=0$, with some constant $c$. It should be easy to see that this force is a vector that points from the current position to the goal position, so the ball will move towards the goal. This right here is a proportional controller! 

Obviously if is the constant $c$ is too high the ball will overshoot the goal (blow dried at max power), and if c is too low, the ball will move very slowly towards the goal (if you're trying to push a bowling ball with a blow drier). There are [heuristics](https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method) to tune $c$, but that's out of the scope of this post.  

-------------------

## The problem with existing models

Now, where do variational models come in? 

If we are trying to learn a representation of the ball system just from videos, a typical approach involves learning a mapping from pixels to ball state, $x_t = \text{encoder}(I_t)$, and a mapping from the current state to the next state given an action, $x_{t+1} = \text{transition}(x_t, a_t)$. This is the general idea behind variational models like E2C and PlaNet. With these mappings, it is possible to do __model-predictive control (MPC)__. In MPC from images, we start by inferring the current state of the system using the encoder; we then find a sequence of actions by minimizing the distance to the goal state after N steps (which we can simulate using the transition function). We choose and perform the best action, and repeat the process given the next frame from the enviornment. Common MPC algorithms include [LQR](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) and [CEM](https://en.wikipedia.org/wiki/Cross-entropy_method).

The disadvantage of this approach is that it is computationally expensive and requires learning a transition function. In contrast, proportional controllers are much simpler and model free, and thus do not require knowledge of the transition function to perform control. This makes them the simplest and most common type of controller used in industrial applications. It would be very cool if we could use them to perform control on learned variational models. 

How do existing variational models behave under a proportional controller? 

As a starting point for our investigation, we trained an [E2C](https://arxiv.org/abs/1506.07365) model on random transitions from a simulated 2D actuated ball moving in the horizontal plane. The E2C model learns a locally linear transition model and is known for finding interpretable state representations (i.e. it learns state dimensions corresponding to the position and velocity variables), and it was one of the first deep variational models that built additional structure into the transition function in order to learn better state representations. Having trained the model, we apply actions of the form 

$$a_t \propto (x^{\text{goal}} - x_t)$$

 for a random goal and initial position. The resulting trajectory in latent space can be seen below.

{:refdef: style="text-align: center;"}
![e2c-trajectory](/assets/figures/newtonianvae/pointmass_e2c_example_path.png){:width="30%"}
{: refdef}

We can see that applying a force in the direction of the goal state does not move the ball towards the goal state! After some thought, we realized that this is because E2C learns a transition model that is too unconstrained to perform proportional control, since its transition model has the form:

$$ x_{t+1} = A x_t + B a_t $$

In the next section we will figure out what constraints to impose on $A$, $B$ and $C$ in order to allow proportional control. We will need to formulate a transition model that more accurately describes the newtonian relationship between position, velocity and force.

One additional problem with most variational models (like [E2C](https://arxiv.org/abs/1506.07365), [DVBF](https://arxiv.org/abs/1605.06432) or [DKF](https://arxiv.org/abs/1511.05121)) is that they don't have explicit position and velocity latent variables by construction. Typically a latent vector z is learned which ends up containing x and v in it, but identifying which dimensions of z corresponds to which is has to be made post-training, which is impractical.  

----------------------------------------------

## Our model: the NewtonianVAE

Our formulation is as follows. Writing Newton's equations in a locally linear form we have:

$$
\begin{aligned} 
\frac{dx}{dt} &= v \\
\frac{dv}{dt} &= A(x,v) x + B(x,v) v + C(x,v) u
\end{aligned}
$$

The $A$ term models spring-like behaviour, the $B$ term models friction, and the $C$ term models the external force/action.

To build this into a variational model we use the static system configuration $x$ (positions/angles) as the stochastic variable that is inferred by the approximate posterior, $x_t \sim q(x_t\|I_t)$, and velocity is simply the discreve derivative of inferred positions, $v_t = (x_t-x_{t-1})/\Delta t$.

With this construction we define the generative model as:

joint distribution | $p(I_{1:T}, x_{1:T}\| u_{1:T}) = p(I_{1:T}\| x_{1:T}, u_{1:T}) p(x_{1:T}\| u_{1:T})$
observation likelihood | $p(I_{1:T}\| x_{1:T}, u_{1:T}) = \prod p(I_t\| x_t)$ 
transition prior | $p(x_{1:T}\| u_{1:T}) = \prod p(x_t\|x_{t-1}, u_{t-1}; v_t)$

where we write the transition prior in the locally linear form:

$$ 
\begin{aligned} 
p(x_t|x_{t-1}, u_{t-1}; v_t) &= N(x_t|x_{t-1} + \Delta t \cdot v_t, \sigma^2) \\
v_t &= v_{t-1} + \Delta t \cdot (A x_{t-1} + B v_{t-1} + C u_{t-1})
\end{aligned} 
$$

with $[A, log(-B), log(C)] = \text{diag}(f(x_t, v_t, u_t))$, where $f$ is a neural network with linear output. During inference, $v$ is computed simply as:

$$ v_t = (x_t-x_{t-1})/ \Delta t $$ 

where $x_t \sim q(x_t\|I_t)$ and $x_{t-1} \sim q(x_{t-1}\|I_{t-1})$. Both the reconstruction and posterior terms are parameterized by a neural network.

Notice that we simply built a discrete form of into the transition model (3). However, the key idea that will make this work as intended (that is, that will allow proportional controllability), is the fact that we use 
1. diagonal matrices $A$, $B$ and $C$;
2. non-negative $C$.

Diagonality encourages correct coordinate relations between $u$, $x$ and $v$, since linear combinations of dimensions are eliminated. Using non-negative $C$ is key to obtain a correct directional relation between the actions and the states (suppose $C$ is negative; now if you pointed the blow the drier towards the goal the ball would move backwards!). We use strictly negative B to provide a correct interpretation of the v term as friction, which adds trajectory stability.

We train all the components end-to-end on a dataset of random transitions by minimizing the lower bound on the marginal likelihood (ELBO):

$$
L = \mathbb{E}_{q(x_t | I_t)q(x_{t-1} | I_{t-1})} \left[ \mathbb{E}_{p(x_{t+1}|x_t, u_t; v_t)}  p(I_{t+1}|x_{t+1}) + \text{KL}\left(q(x_{t+1}|I_{t+1}) \|  p(x_{t+1}|x_t, u_t; v_t) \right) \right]
$$

For more details and derivations refer to the full [paper](https://arxiv.org/abs/2006.01959).



--------------------

## Some results

So... with the maths out of the way, let's see what we can actually do with this.

First, let's see how our model behaves under proportional control, like we did with E2C before. We will compare the __NewtonianVAE__ (formulation above), the __Full-NewotnianVAE__ (formulation above but with full matrices $A$, $B$ and $C$), the __E2C__ model, and a standard __VAE__ trained on individual frames. Using the point mass environment as before, and a 2-arm reacher environment (both adapted from the [dm_control](https://github.com/deepmind/dm_control) library), we train the models on a dataset of random transitions. Once trained, applying a proportional controller as before we get the following trajectories:

{:refdef: style="text-align: center;"}
![trajectories](/assets/figures/newtonianvae/main_sample_trajectories_tight.png){:width="65%"}
{: refdef}

We can see that only the NewtonianVAE produces P-controllable latent states, as all the remaining models fail to reach the goal under a P-controller. More importantly, we see that the diagonality and sign constraints imposed on $A$, $B$ and $C$ are essential in order to learn a latent space that is correct in the physical sense.
We can get further insight by inspecting the latent spaces learned by each model:

{:refdef: style="text-align: center;"}
![latent-spaces](/assets/figures/newtonianvae/all_latent_spaces.png){:width="65%"}
{: refdef}

We see that only the NewtonianVAE was able to learn a representation aligned with the control coordinates: $[x,y]$ in the point mass, $[\theta_1, \theta_2]$ in the reacher. It also highlights the fact that even though the latent spaces learned by the Full-NewtonianVAE and E2C are seemingly well structured for the point mass system, they fail to provide P-controllability. While these systems <u>can</u> still be controlled using model-predictive control (such as LQR or CEM), this is unnecessary with a P-controllable latent space.


-------------------

OK so this is exciting, but what else can we do with this? Well, it turns out that if you can control your system with proportional controllers, you can apply that to __imitation learning__ to do automated sub-goal discovery and task segmentation!

{:refdef: style="text-align: center;"}
![latent-spaces](/assets/figures/newtonianvae/reacher_demo.gif){:width="15%"}
{: refdef}

To show this, we devised a simple task where the 2-arm reacher has to reach for 3 colored objects in succession (shown above). Given a visual demonstration sequence D we encode the frames using the inference network $q(x\|I)$ and obtain demonstration in our latent space, D. We then fit a __mixture density network (MDN)__ of P-controllers with the form:

$$ p(u_t|x_t) = \sum_n \pi_n(x_t) \mathcal{N}(u_t|K_n(x^{goal}_n - x_t), \sigma^2) $$

where $K_n$, $x^{goal}_n$ and $\sigma$ are learnable parameters, and $\pi(x)$ is some parametric function - in our case, a simple linear layer sufficed. Intuitively, fitting this MDN to the demonstrations will allow $\pi_n(x_t)$ to split the trajectories into regions, with a single controller $n$ being responsible for each region.

If we train a 3-component MDN on <u>a single demonstration trajectory</u> we get the following plot:

{:refdef: style="text-align: center;"}
![latent-spaces](/assets/figures/newtonianvae/learned_mdn.png){:width="40%"}
{: refdef}

The connected white circles show the demonstration trajectory in the learned latent space. Each background color corresponds to an output unit of $\pi(x)$ (indicating the assigned controller), and the diamonds show each learned $x^{goal}_n$ in latent space. We can see that the controller mixture does indeed assign a controller to each sub-task. As an added bonus, since we learned a reconstruction network, $p(I_t\|x_t)$, during training, we can decode the learned sub-goals and visualize them in image space!

{:refdef: style="text-align: center;"}
![latent-spaces](/assets/figures/newtonianvae/decoded_goals.png){:width="30%"}
{: refdef}

------------------------------------

As a proof of concept for real robot applications, we performed a similar object-reaching task using a 7-DoF PR2 robot arm that moves between 6 objects in succession in a hexagon pattern. Since the robot is actuated by a 7-dimensinoal torque along each joint, we use a 7-dimensional configuration vector $x$. We trained the NetwotnianVAE on 700 frames and then learned the mixture of 6 P-controllers on a demonstration sequence of 100 frames. 

{:refdef: style="text-align: center;"}
![latent-spaces](/assets/figures/newtonianvae/real_reacher_demo.gif){:width="15%"}
{: refdef}

Each color in the animation above corresponds to an active controller. The animation above shows that the sub-tasks are correctly identified (moving from each object to the next), which is a promising sign that our model can be used in real-world environments. We also note that the model is able to find the correct segmentations even though not all of the joints are visible in every frame.

Since we used so little data to train the NewtonianVAE, we had to impose further constraints to the structure, which are described in the [paper](https://arxiv.org/abs/2006.01959).

-------------------------------------

## Thoughts and Takeaways

Building physical constraints into deep learning models has seen increased interest in the last couple years. I see this is as an important research direction as we move away from fully black box systems with uninterpretable latent variables and towards interpretable systems. Any problem where there is an interest in inspecting or manipulating physical quantities (be it positions, velocities, material properties, fluid flow, etc.) will lend itself to the integration of physics and deep learning.

I see the NewtonianVAE as a step in the direction of using physical knowledge to simplify robot dynamics representation at the latent level, in order to drastically simplify control of downstream tasks. Of particular interest is the fact that when using a physical latent representation, we can do robust imitation learning using a single demonstration sequence, which is a massive improvement in data efficiency when compared to other methods. 

Interesting future directions (feel free to contact me if you want to tackle any of these :D) include:
	- Currently the model has to be trained on the scene with the goals already in place. That is, we can't move the colored objects around at test time because that would fall outside of the visual domain seen by the encoder during training. It would greatly increase the generality of this approach if new objects at different locations could be handled at test time;
	- Allowing for a varying or distracting background, while still correctly modelling the foreground object of interest;
	- Integrating the NewtonianVAE with an object-centric segmentation approach in order to model multiple objects, possibly interacting with each other.

For questions email me at <small>m dot a dot m dot jaques at sms.ed.ac.uk</small>.

------------------------------------

Many thanks to André Melo ([Twitter](https://twitter.com/_andrenmelo)) for proofreading and suggestions.






