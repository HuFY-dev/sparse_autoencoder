# Modified Loss, Activation, Decoder Arch, and Here is WHY

## Theory

The original architecture of sparse autoencoder (SAE) is: 

$$x' = \text{SAE}(x) = W_d f(x)+b_d$$

where

$$f(x) = \text{ReLU}(W_e x + b_e)$$

and $W_d$ is constrained to be column-normal for optimization purposes.

The loss is calculated as follows:

$$L(x',x)=\alpha||f(x)||_1+||x'-x||_2$$

where $||\cdot||_1$ and $||\cdot||_2$ represents $L_1$ and $L_2$ norms, respectively

Now let's do a theoretical analysis on this loss.

Define $\hat{x}=x/||x||_2$, which is the normalized $x$, then we have

$$||f(x)||_1=||\text{ReLU}(W_e x + b_e)||_1=||x||_2\cdot||\text{ReLU}(W_e \hat{x} + b_e')||_1$$

where $b_e'=b_e/||x||_2$.

Similarly,

$$||x'-x||_2=||x'||_2^2+||x||_2^2-2x'\cdot x=||x||_2^2(\frac{||x'||_2^2}{||x||_2^2}+1-\frac{2x'\cdot x}{||x||_2^2})$$

At first glance, this might not be obvious, but if our reconstruction $x'$ is similar enough to $x$,
we can take $||x'||_2\approx||x||_2$ and the equation simplyfies to 

$$||x'-x||_2=||x||_2^2(1+1-2\frac{x'\cdot x}{||x||_2\cdot||x'||_2})=2||x||_2^2(1-cos(x',x))$$

Now we can rewrite our loss:

$$L(x',x)=\alpha||x||_2\cdot||\text{ReLU}(W_e \hat{x} + b_e')||_1+2||x||_2^2(1-cos(x',x))$$

Notice that, if we mainly care about the direction of $x'$ and the $L_0$ norm as the measure of
sparsity, then the $(1-cos(x',x))$ and $||\text{ReLU}(W_e \hat{x} + b_e')||_1$ terms are more
reflective, while $||x||_2$ serves as a bias term towards high norm tokens. In other words, tokens
with higher norms in a layer generally get higher losses, and the SAEs fits them better than tokens
with lower norms.

## Observations in LLMs

Now, you may ask, does this really matter?

In fact, it does. Firstly, the norm of residuals are NOT from uniform in large language models
(LLMs). Here is a plot showing the norm distribution of residual states across the 12 layers of
GPT2-small:

<center>
    <img src="./images/GPT2norms.png" alt="Distribution of GPT2 residual stream norms across layers"
    width="400">
</center>

Clearly you can notice that, within a single layer the norms are vastly different, and these norms
also vary across layers.

The consequence of this effect is twofolds:

1. Within one layer, SAE will bias residuals with high norms, especially when norms grows above $10^2$.
2. Across layers, since the $L_1$ loss has a factor of $||x||_2$ while the $L_2$ loss has a factor
   of $||x||_2^2$, the change in norm will bias towards one of the loss terms, effectively making $\alpha$
   different across layers.

## Experiments with SAEs

I started with investigating the feature activation corresponding to the top 1% residual states in
terms of their norms. Here are the $L_0$ and $L_1$ norms of the feature activations:

<center>
    <img src=".\images\feature_act_l0.png" alt="L0 norm of feature activations" width=600>
</center>

<center>
    <img src=".\images\feature_act_l1.png" alt="L1 norm of feature activations" width=600>
</center>

This does not look like things are working properly :(

It turns out, these high norm tokens corresponds to residual states from the `<|endoftext|>` token.
They are pretty uninterpretable and are less monosemantic. Hence, it might be beneficial to reduce
the loss of these high norm tokens to allow the model to pick up more information from other tokens.

## Quick fix

We can change the model architecture a bit.

Define the **Normalized SAE**

$$\text{SAE}_{\text{N}}(x)=W_d(\text{act}(W_ex+b_e))+b_d$$

where we no longer constrain $W_d$ to be column normal and let
$\text{act}(x)=\text{tanh}(\text{ReLU}(x))$, then the activation looks like this:

<center>
    <img src=".\images\activation.png" alt="Plot of the activation function" width=400>
</center>

The purpose of this is to constrain feature activations to the range $(0,1)$, so the $L_1$
norm will not explode for high-norm residuals. We can still learn the reconstruction with a non-unit
norm decoder, and use it as a feature dictionary that contains information on the norm of residual
states.

Now we have dealt with the $L_1$ loss, what about the $L_2$ term? This can be more complicated.
Depending on our goals, we can either choose to divide the $L_2$ loss by $||x||_2^2$, or leave it as
is. The prior eliminates the bias towards high-norm residuals, while the latter assumes residual
states with higher norms are more important than those of lower norms. There is also a third option
to normalize the $L_2$ by dividing with $||x||_2$, which leaves $L_2$ to grow linearly with
$||x||_2$ and prevents it from exploding as norms grows larger than $10^2$.

## Training

TODO.