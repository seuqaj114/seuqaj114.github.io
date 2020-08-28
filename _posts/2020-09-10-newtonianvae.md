---
title:  "NewtonianVAE"
layout: post
---

This blog post will explain the ideas behind my most recent [paper](https://arxiv.org/abs/2006.01959), _NewtonianVAE: Proportional Control and Goal Identification from Pixels via Physical Latent Spaces_, and provide some ideas for future work that either me (or you!) could tackle.

Before diving into the paper itself, I will quickly explain the concept of PID and model-based control for those of you who are not familiar with this area.

--------------------

## PID control

PID stands for Proportional-Integral-Derivative, which is a mouthful... Let's ignore the I and D terms, and focus just on P, as it is the most important one. 
So what is a proportional controller (P-controller)? Let's imagine a very simple system like a ball rolling on a horizontal table, to which we can apply a force along the horizontal plane (for example, by pushing it with a blow drier). This sistem is described by the 2D position of the ball p = [x,y] and the action is the force applied along this plane, F = [F_x, F_y]. 

If the ball is initially at a position p_0 = [x_0, v_0] and we want to move it towards a goal position p_g = [x_g, y_g], a simple way to achieve this is by applying, at each time t, a force of the form F_t = c (p_g - p_t), starting at t=0, with some constant c. It should be easy to see that this force is a vector that points from the current position to the goal position, so the ball will move towards the goal. This right here is a proportional controller! 

Obviously if is the constant c is too high the ball will overshoot the goal (blow dried at max power), and if c is too low, the ball will move very slowly towards the goal (if you're trying to push a bowling ball with a blow drier). There are heuristics to tune c, but that's out of the scope of this post.  

-------------------

Now, what does this have to do with VAE's? 

If we are trying to learn a representation of the ball system just from videos, a typicall approach involves learning a mapping from pixels to ball state, s_t = enc(img_t), and a mapping from the current state to the next state given an action, s_{t+1} = transition(s_t, a_t). This is the general idea behind variational models like E2C and PlaNet. With this mapping, it is possible to do model-predictive control, where we choose a sequence of actions that will lead the system to the goal state (which we can approximately know since we have the transition function). Once an action is performed, we can do this process again to correct for disparities between the learned transition model and the real system's evolution. Algorithms for finding the best sequence of actions given a transition model include LQR, iLQR and CEM.

Proportional controllers are the most ubiquitous and simple form of controller (and they are model free, since we don't need to know the transition function), so it would be very cool if we could use them to perform control on learned variational models.


How do existing variational models behave under a proportional controller? (jump 1)

As a starting point for our investigation, we started by learning an E2C model on a simulated 2D ball system. The E2C model learns a locally linear transition model and is known for finding interpretable state representations (i.e. it learns state dimensions corresponding to the position and velocity variables), ...

We can see that applying a force in the direction of the goal state does not move the ball towards the goal state! After some thought, we realized that this is because E2C learns a transition model that is too unconstrained in order to learn states that are physically plausible. By physically plausible, we mean that an action applied along some direction should move the system in that direction. Since the transition model of E2C has the form p_{t+1} = A p_t + B a_t, we can violate the physical plausibility by simply making B a negative matrix or a non-diagonal matrix. In order to obtain physical plausibility, and hence the ability to perform proportional control, we need to formulate a transition model that more accurately describes the newtonian relationship between position, velocity and force.

One additional problem with most variational models (like EC2, DVBF or DKF) is that they don't have explicit position and velocity latent variables by construction. Typically a latent vector z is learned which ends up containing x and v in it, but identifying which dimensions of z corresponds to which is has to be made post-training, which is impractical.  

Our formulation is as follows. Writing Newton's equations in a locally linear form we have:

$$ \frac{dx}{dt} = v \quad , \quad \frac{dv}{dt} = A(x,v) x + B(x,v) v + C(x,v) u$$

A term in x models spring-like behaviour, a term in v models friction, and the term in u models the external force/action.

To turn build this into a variational model we use the static system configuration (positions/angles) x as the stochastic variables that is inferred by the approximate posterior, x_t \sim q(x_t|I_t), and velocity is simply the difference of inferred positions, v_t = (x_t-x_{t-1})/\Delta t.

With this construction we define the generative model as:

p(I_{1:T}| x_{1:T}, u_{1:T}) = \prod p(I_t| x_t)  -> observation likelihood term (images are conditionally independent given the respective latent states)
p(x_{1:T}| u_{1:T}) = \prod p(x_t|x_{t-1}, u_{t-1}; v_t) -> transition prior

where we write the transition prior in the locally linear form:
p(x_t|x_{t-1}, u_{t-1}; v_t) = \mathcal{N}(x_t|x_{t-1} + \Delta t \cdot v_t, \sigma^2) (3)
v_t = v_{t-1} + \Delta t (A x_{t-1} + B v_{t-1} + C u_{t-1})

with [A, log(-B), log(C)] = diag(f(x_t, v_t, u_t)), where f is a neural network with linear output. During inference, v is computed simply as the difference of x's:
v_t = (x_t-x_{t-1})/\Delta t, where x_t \sim q(x_t|I_t) and x_{t-1} \sim q(x_{t-1}|I_{t-1}). Both the reconstruction and posterior terms are parameterized by a neural network.

Notice that we simply built a discrete form of into the transition model (3). However, the key idea that will make this work as intended (that is, that will allow proportional controllability), is the fact that we use diagonal matrices A, B, C. This encourages correct coordinate relations between u, x and v, since linear combinations of dimensions are eliminated. Using also strictly positive C is key to obtain a correct directional relation between the actions and the states (otherwise you could blow the drier towards the goal and the ball would move backwards :/). We use strictly negative B to provide a correct interpretation of the v term as friction, which adds trajectory stability.

We train all the components end-to-end by minimizing the ELBO:
....

For more details on the derivation please see Appendix B in our paper.

This formulation is what we call the NewtonianVAE.

--------------------

Sooooo, with the maths out of the way, let's see what we can actually do with this.

First, let's see how our model behaves under proportional control, like we did with E2C before. Using the point mass environment as before, and a 2-arm reacher environment (both adapted from the dm_control library), we train a NewtonianVAE on a dataset of random transitions. Once trained, apply a proportional controller as before and we get the trajectories below:

(Figure 3 here)

We can see that only the NewtonianVAE produces P-controllable latent states, as all the remaining models fail to reach the goal under a P-controller.
We can get more insight into this by inspecting the latent spaces learned by each model:

(Figure 2 here)

We see that only the NewtonianVAE was able to learn a representation aligned with the control coordinates ([x,y] in the point mass, [theta1, theta2] in the reacher). It also highlights the fact that even though the latent spaces learned by the Full-NewtonianVAE and E2C are seemingly well structured for the point mass system, they fail to provide P-controllability. While these systems can still be controlled using model-predictive control, this is unnecessary with a P-controllable latent space.


-------------------

OK so this is exciting, but what else can we do. Well, it turns out that if you can control your system with proportional controllers, you can apply that to imitation learning to do automated sub-goal discovery!

To show this, we constructed a simple task where the 2-arm reacher has to reach for 3 colored objects in succession. Given a visual demonstration sequence D we encode the frames using the inference network q(x|I) and obtain demonstration in our latent space, D. We can then fit a mixture of P-controllers to the demonstrations:

p(u_t|x_t) = \sum_n \pi_n(x_t) \mathcal{N}(u_t|K_n(x^{goal}_n - x_t), \sigma^2)

where K_n, x^{goal}_n and \sigma are learnable parameters, and \pi(x) is a linear function. Intuitively, fitting this mixture to the demonstrations will split the trajectories into regions, with a single controller being responsible for each region.

If we do this _with a single demonstration trajectory_ we get

(Figure 5 top, left)

We can see that the controller mixture does indeed assign a controller to each sub-task. As an added bonus, since we have x^{goal}_n and we trained a decoder p(I_t|x_t) earlier, we can decode the learned sub-goals and visualize them in image space, to verify that they match the goal true states!

(Figure 5 bottom, left)



As a proof of concept for real robot applications, we performed a similar object-reaching task using a 7-DoF PR2 robot arm that moves between 6 objects in succession in a hexagon pattern. Training the NetwotnianVAE on 700 frames and then training the mixture of P-controllers on a demonstration sequence of 100 frames we get:


## Takeaways

Building physical constraints into deep learning models has seen increased interest in the last couple years. I see this is as an important research direction as we move away from fully black box systems with uninterpretable latent variables and towards interpretable systems. Any problem where there is an interest in inspecting or manipulating physical quantities (be it positions, velocities, material properties, fluid flow, etc.) will lend itself to the integration of physics and deep learning.

I see the NewtonianVAE as a step in the direction of using physical knowledge to simplify robot dynamics representation at the latent level, in order to drastically simplify control of downstream tasks. Of particular interest is the fact that when using a physical latent representation, we can do robust imitation learning using a single demonstration sequence, which is a massive improvement in data efficiency when compared to other methods. 

Interesting future directions (feel free to contact me if you want to tackle any of these :D) include:
	- Currently the model has to be trained on the scene with the goals already in place. That is, I can't move the colored objects around at test time because that would fall outside of the visual domain seen by the encoder during training. It would greatly increase the generality of this approach if new objects at different locations could be handled at test time;
	- Allowing for a varying or distracting background, while still correctly modelling the foreground object of interest;
	- Integrating the NewtonianVAE with an object-centric segmentation approach in order to model multiple objects, possibly interacting with each other.

The code and pretrained models is available in this [repo](https://github.com/seuqaj114/newtonianvae.git)






