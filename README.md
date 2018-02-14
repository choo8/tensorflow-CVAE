# tensorflow-CVAE
Implementation of the Conditional Variational Auto-Encoder (CVAE) in Tensorflow

While learning more about CVAEs, I decided to attempt to replicate some of the results from the paper "Semi-Supervised Learning with Deep Generative Models" by Kingma et al. to internalize my learning. The code implementation is referenced from the code and papers below. If you find any typos or mistakes in my code, please let me know!

# M1 Model
To train the M1 VAE model, you can run `python train_M1.py -train`. If you wish to use the trained weights, just leave out the train flag and run `python train_M1.py`. Here are some results:

![labeling](https://github.com/choo8/tensorflow-CVAE/blob/master/M1_model_100.gif)

# M2 Model
Similar to the M1 VAE model, you can run `python train_M2.py -train` to train the M2 CVAE model. This implementation is the stacked M1+M2 model as described in the original paper. If you wish to use the trained weights, just leave out the train flag and run `python train_M2.py`. Note that since this is the stacked M1+M2 model, the trained weights for M1 are required for. The MNIST analogies can also be obtained by running `python run_analogy.py`. Here are some results:

![labeling](https://github.com/choo8/tensorflow-CVAE/blob/master/temp/analogy.png)

# Conclusions
I was not able to obtain the 96% accuracy using 100 labelled data points and 49900 unlabelled data points as described in the paper. However, I was able to obtain 90% accuracy with the stacked M1+M2 model after 1000 epochs. I may be able to obtain higher accuracy values but I did not continue the training. Also, I have experimented with Batch Normalization for both models but it only seemed to worsen the results, so I did not upload the implementattion of M2 with Batch Normalization.

The MNIST analogies did not look very good, there could be more experimenting by inputting the image data directly into M2 instead of using the latent representation obtained from M1.

## References
[1] [Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298) by Kingma et al. </br>
[2] https://github.com/saemundsson/semisupervised_vae </br>
[3] https://github.com/Gordonjo/generativeSSL </br>
[4] https://github.com/musyoku/variational-autoencoder
