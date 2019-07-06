# InfoGAN

This is a pytorch implementation of [InfoGAN](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets).

This repository has the following features that others do not have:

- Highly customizable.
  - You can use this for your own dataset, settings.
  - Most parameters including the latent variable design can be customized by editing the yaml config file.

- OK clean, structured codes
  - This is totally my personal point of view. :wink:

- [TensorBoard](https://www.tensorflow.org/tensorboard) is available by default.

## Result

  **COMING SOON**

## Prerequisites

- Python (`~3.6`)

## Getting Started

```sh
make setup
```

- Start training

  ```sh
  python src/train.py --config <config.yaml>
  ```

  You need to specify all of training settings with `yaml` fromat. Example files are placed under `configs/`.

  If you want to try training anyway, my configuration for debugging is available.

  ```sh
  make debug
  ```

- Open tensorboard

  ```sh
  make tb
  ```
  Training metrics (ex. loss) are printed on console and tensorboard.

  By default, tensorboard watches `./results` directory. To change the path, execute `tensorboard --logdir <path>` or edit `Makefile`.


## TODO

- [ ] upload results on MNIST, Fashion-MNIST datasets.
- [ ] automatic hyper-parameters tuning with Optuna.

- [ ] separate `z` variable and `c` variable from `latent_vars`.
