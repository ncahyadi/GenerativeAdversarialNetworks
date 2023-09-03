# Wasserstein GAN (WGAN)

Wasserstein GAN, introduced in 2017 by Arjovsky et al. [source](https://arxiv.org/abs/1701.07875), is a variant of Generative Adversarial Networks (GANs) that offers several advantages for training. It addresses issues such as mode collapse and vanishing gradients.

## Key Concepts

### Wasserstein Loss

Wasserstein loss is a new loss function used in WGANs that measures the Wasserstein distance. Wasserstein distance quantifies the effort needed to transform one probability distribution into another.

### Critic (Discriminator)

In WGAN, the discriminator is referred to as the critic. The critic must satisfy the Lipschitz constraint, ensuring Lipschitz continuity.

### Lipschitz Constraint

WGAN enforces the Lipschitz constraint, specifically the 1-Lipschitz constraint, using weight clipping. This constraint helps stabilize training.

## Mode Collapse

Mode collapse occurs when a GAN generates a limited variety of images, typically from a single class, even if it was trained with diverse input data. Each iteration of the generator tends to overfit to a specific critic, preventing it from exploring a broader range of output.

## Vanishing Gradients

In some cases, when the critic becomes too effective, it can lead to generator training failures. The critic outputs values between 0 and 1, making it challenging for the generator to learn effectively. This results in a strong critic but a weak generator.

## Benefits of Wasserstein Loss

Wasserstein loss addresses the vanishing gradients problem and mitigates mode collapse. It allows the critic to optimize without concerns about vanishing gradients, thereby encouraging the generator to explore new possibilities.

Wasserstein loss is used to approximate the Earth Mover's Distance (EMD), which measures the effort required to match one distribution to another. Unlike traditional GAN losses that focus on fooling the critic, WGAN's loss function reflects the quality of generated images, making it easier to evaluate image quality.

## Common Distance Metrics

In traditional GANs, common distance metrics include Kullback-Leibler (KL) divergence, Jensen-Shannon (JS) divergence, and Wasserstein distance. JS divergence is often used but can present gradient-related problems, leading to the adoption of Wasserstein distance in WGAN.

## WGAN Function

The loss in WGAN is the difference between the expected values of the critic's output for authentic and generated images. The critic aims to maximize this difference, while the generator seeks to minimize it.

## Challenges with WGAN

1. **Lipschitz Constraint**: Although weight clipping enforces the Lipschitz constraint for Wasserstein distance calculation, it can generate issues such as low-quality images and convergence problems, particularly when the hyperparameter 'c' is not adjusted correctly.

2. **Model Capacity**: Weight clipping acts as a form of weight regularization, reducing the model's capacity and limiting its ability to represent complex functions. Enhanced variants like WGAN-GP address this limitation by providing smoother training and complex boundary modeling.

## WGAN-GP (Wasserstein GAN with Gradient Penalty)

To overcome challenges in WGAN, a variant called WGAN-GP was proposed by Gulrajani et al. in [source](https://arxiv.org/abs/1704.00028). Instead of relying on weight clipping, WGAN-GP uses a gradient penalty approach. This approach considers the gradient norm of the discriminator, considering its input, to enforce the Lipschitz constraint.

WGAN-GP effectively solves previous issues and offers stable training for various GAN architectures. It replaces weight clipping with gradient penalty, ensuring that the L2 norm of the gradient remains close to 1 during training.

For more detailed information and implementation examples, please refer to the provided sources.
