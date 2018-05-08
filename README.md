## Implementing a Generative Adversarial Network (GAN/DCGAN) to get new Human Faces
Like the Variable Autoencoder (VAE), the DCGAN is an architecture for learning to generate new content.
And just like the VAE, a DCGAN consists of two parts. In this case, these are:

*The discriminator*, which learns how to distinguish fake from real objects of the type we’d like to create

*The generator*, which creates new content and tries to fool the discriminator

The basic idea is that both network parts compete with each other.
When the discriminator becomes better, the generator needs to become better too,
otherwise it can’t fool the discriminator any longer. Similarly, when the generator becomes better,
the discriminator has to become better also, else it will lose the ability to distinguish fake from real content.

This project uses a Generative Adversarial Network (GAN) trained on celebrity faces to generate new faces that don't appear in the training set. This method for generating faces is really cool
and can actually reach equilibrium fairly quickly when compared to training other types of neural networks, such as LSTM cells.
I first test the GAN on the MNIST dataset to generate new handwritten digits. Then I train the GAN on the CelebA dataset for generating new human faces.

The GAN is built on TensorFlow, written in Python 3 and is presented via Jupyter Notebook. Trained via # FloydHub gpu# instance.

Below is an example of some faces my GAN was able to generate.

![GAN generated faces](/
