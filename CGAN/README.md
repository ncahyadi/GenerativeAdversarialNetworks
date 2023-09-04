# CGAN - Conditional Generative Adversarial Network

**GAN Problems:**

One of the challenges with traditional Generative Adversarial Networks (GANs) is that they operate like a black box. We lack an understanding of how the latent vector mapping corresponds to the actual image features. Consequently, manipulating these features to produce specific results becomes a challenging task. The final image is generated purely from random noise.

To address this limitation, Conditional GAN (CGAN) was introduced.

## Conditional GAN [Read more](https://arxiv.org/abs/1411.1784)

CGAN enables the generation of images based on specific conditions. In this framework:

- The generator and discriminator receive additional information to produce images belonging to specific classes.
- The model learns to map input data and generate output images using various contextual information.
- Conditional data is used to combine noise with labels from a particular class of objects.

**Advantages of CGAN:**

1. **Faster Convergence**: CGANs converge more quickly, and the distribution of generated images follows a consistent pattern.
2. **Control Over Output**: CGANs offer greater control over the generator's output, allowing you to provide specific information and conditions.

Using the MNIST dataset, for instance, traditional GANs cannot generate a specific number because they lack control over the generation process. CGANs, on the other hand, introduce an input layer that guides the generator in producing specific types of images. This conditioning occurs after providing class information to both the discriminator and generator networks.

## Applications:

**1. Image-to-Image Translation:**

   a. **pix2pix:** This method enables paired translation, where it learns to map input images to output images with different properties.

   b. **CycleGAN:** Unlike pix2pix, it performs unpaired translation, allowing training with images from two different domains without needing a paired dataset.

**2. pix2pix Intuition [Read more](https://arxiv.org/abs/1611.07004):**

In pix2pix, the system takes an input image and transforms it to generate an output image with different characteristics. For example, it can convert a drawing into a realistic image or turn a segmented image into a more realistic one. This approach relies on having equivalent pairs of images for training, known as a paired dataset.

**3. pix2pix GAN:**

pix2pix GAN is an approach for training deep convolutional neural networks for paired picture-to-picture translations. It maps the input image 'x' and a random noise vector 'z' to generate the output 'y.' The generator 'G' is trained to produce outputs that are indistinguishable from real images, while the discriminator 'D' is trained to detect fake images as effectively as possible.
