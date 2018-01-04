# TensorFlow LadderEstimator
This project contains a TensorFlow implementation of the Ladder Network. The model is written conform the [Estimator API](https://www.tensorflow.org/programmers_guide/estimators), an easy to use interface provided by TensorFlow.
## Ladder Networks
Ladder Networks are a structure of neural networks that allow to be trained with few labeled data. In other words, Ladder Networks provide a way to do semi-supervised learning with deep neural networks. Given a feedforward network the Ladder Network is constructed by creating the inverse network and requiring to reconstruct the input from the output similar to autoencoders.

The concept of Ladder Networks can be used in all kinds of feedforward neural networks or even recurrent networks. However, in this project only fully connected, convolutional, meanpool and maxpool variants are implemented. Each layer of the encoder part (the feedforward network) consists of four parts:

- noise addition layer
- transformation (fc/conv/pool) layer
- batch normalization (cfr. [S Ioffe, C Szegedy](http://www.jmlr.org/proceedings/papers/v37/ioffe15.html))
- activation 

As the decoder is the encoders counterpart, it consists equivalently of four parts:
- denoise layer (for intuitive explanation: [Learning by Denoising](https://thecuriousaicompany.com/another-test-learning-by-denoising-part-1-what-and-why-of-denoising/)
- inverse transformation(fc/deconv/unpool)
- batch normalization
- activation

The order of layers and their size fully parametrized. This makes them fully customable! However, the convolutional layers are automatically padded with zeros to prevent deconvolution artifacts [Deconvolution and Checkerboard Artifacts - A Odena, V Dumoulin, C Olah](https://distill.pub/2016/deconv-checkerboard/)

A nice intro on Ladder Networks is available on YouTube: 
- <a href="http://www.youtube.com/watch?feature=player_embedded&v=ZlyqNiPFu2s
" target="_blank"><img src="http://img.youtube.com/vi/ZlyqNiPFu2s/0.jpg" 
alt="Symposium: Deep Learning - Harri Valpola" width="240" height="180" border="10" /></a>

If you want to dive deeper in the subject I would suggest you to read: 
- [Semi-Supervised Learning with Ladder Networks- A Rasmus, M Berglund, M Honkala, H Valpola, T Raiko](http://papers.nips.cc/paper/5947-semi-supervised-learning-with-ladder-networks.pdf)
- [Lateral connections in denoising autoencoders support supervised learning - A Rasmus, H Valpola, T Raiko](https://arxiv.org/pdf/1504.08215.pdf)
- [Denoising autoencoder with modulated lateral connections learns invariant representations of natural images - A Rasmus, T Raiko, H Valpola ](https://arxiv.org/pdf/1412.7210.pdf)

