# InfoGAN

This is a pytorch implementation of [InfoGAN](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets).

I decided to make this repository public by following reasons.

- This includes cleaner codes and architecture than others.
  - This is totally my personal point of view. :wink:

- The codes are highly customizable, so you can use more generically.
  - Most parameters including the latent variable design can be customized by editing the yaml config file.

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
