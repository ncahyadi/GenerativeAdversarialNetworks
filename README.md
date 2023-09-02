# GenerativeAdversarialNetworks
# Generative Adversarial Networks (GANs)

## Introduction
- **GANs** (Generative Adversarial Networks) were introduced by Ian Goodfellow and a team from the University of Montreal in 2014.
- They have the remarkable ability to generate entirely new content that has never been seen before.

## Components
- GANs consist of two main components:
  - **Generator**: Responsible for creating new instances of data.
  - **Discriminator**: Evaluates the authenticity of the generated data.

## Working Process
1. The generator starts with random values to generate an image (input).
2. The generated image is then passed to the discriminator along with a stream of images from the real dataset (used as reference).
3. The discriminator returns a probability indicating how real or fake the generated image is, typically ranging from 0 (fake) to 1 (authentic).

## Key Aspects
- The discriminator is a standard convolutional network that performs binary classification (real or fake images).
- In contrast, the generator takes random noise as input and upsamples it to create an image, which is analyzed by the discriminator.
- Both the discriminator and the generator work to optimize different and opposing loss functions.
- During training, their behaviors evolve, often in opposition to each other due to the loss functions.

## Revolutionary Impact
- GANs revolutionized the field of generative modeling by achieving high-quality results on a wide range of datasets.

## Types of GANs
- DCGANs (Deep Convolutional GAN)
- WGANs (Wasserstein GAN)
- SRGANs (Super Resolution)
- Pix2Pix (Image-to-Image)
- CycleGAN (Cycle Generative)
- StackGAN (Stacked GAN)
- ProGAN (Progressive Growing)
- StyleGAN (Style-Based GAN)
- VQGAN (Vector Quantized GAN)
- SGAN
- InfoGAN
- SAGAN
- AC-GAN
- GauGAN
- GFP-GAN

## Applications of GANs
- Paired image-to-image translation [Read More](https://arxiv.org/abs/1611.07004)
- Unpaired image-to-image translation [Read More](https://arxiv.org/abs/1703.10593)
- Super resolution (upsampling low-resolution images) [Read More](https://arxiv.org/abs/1809.00219)
- Text-to-Image Generation (text2image) [Read More](https://arxiv.org/abs/1612.03242)
- Facial rejuvenation and aging [Read More](https://arxiv.org/pdf/1702.01983.pdf)
- Generation of fictional faces
- Data Augmentation (synthesizing new class-specific images) [Read More](https://arxiv.org/abs/1809.11096)
- Image inpainting (filling missing parts of images)
- Style Transfer
- Generating novel human poses [Read More](https://arxiv.org/pdf/2001.01259v1.pdf)
- Restoration of old images and noise removal [Read More](https://arxiv.org/abs/2101.04061)
- 3D Object Generation
- Creation of new anime and Pokémon characters (Illustration GAN and PokeGAN)
- Deep Dream
- DALL·E
- Stable Diffusion
- Midjourney
