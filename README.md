# InfoGAN

This is a pytorch implementation of [InfoGAN](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets).

This repository has the following features that others do not have:

- Highly customizable.
  - You can use this for your own dataset, settings.
  - Most parameters including the latent variable design can be customized by editing the yaml config file.

- OK clean, structured codes
  - This is totally my personal point of view. :wink:

- Tensorborad is available by default.


## Prerequisities

- Python (`~3.6`)

## Getting Started

```sh
make devenv
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
  Training metrics (ex. loss) are printed on console and [TensorBoard](https://www.tensorflow.org/tensorboard).

  By default, tensorboard watches `result` directory. To change the path, execute `tensorboard --logdir <path>` or edit `Makefile`
