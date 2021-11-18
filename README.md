# Generative Adversarial Imitation Learning with PyTorch

This repository is for a simple implementation of Generative Adversarial Imitation Learning (GAIL) with PyTorch. This implementation is based on the original GAIL paper ([link](https://arxiv.org/abs/1606.03476)), and my Reinforcement Learning Collection repository ([link](https://github.com/hcnoh/rl-collection-pytorch)).

In this repository, [OpenAI Gym](https://gym.openai.com/) environments such as `CartPole-v0`, `Pendulum-v0`, and `BipedalWalker-v3` are used. You need to install them before running this repository.

*Note*: The environment's names could be different depending on the version of OpenAI Gym.

## Install Dependencies
1. Install Python 3.
2. Install the Python packages in `requirements.txt`. If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:

    ```bash
    $ pip install -r requirements.txt
    ```

3. Install other packages to run OpenAI Gym environments. These are dependent on the development setting of your machine.
4. Install PyTorch. The version of PyTorch should be greater or equal than 1.7.0.

## Training and Running
1. Modify `config.json` as your machine setting.
2. Execute training process by `train.py`. An example of usage for `train.py` are following:

    ```bash
    $ python train.py --env_name=BipedalWalker-v3
    ```

    The following bash command will help you:

    ```bash
    $ python train.py -h
    ```

## The results of CartPole environment

![](/assets/img/README/README_2021-02-23-23-47-07.png)

## The results of Pendulum environment

![](/assets/img/README/README_2021-02-27-22-24-07.png)

## The results of BipedalWalker environment

![](/assets/img/README/README_2021-02-23-23-48-04.png)

- This result suggests that the causal entropy has little effect on the performance.

## Recent Works
- The CUDA usage is provided now.
- Modified some errors in GAE.
- Modified some errors about horizon was corrected.

## Future Works
- Search other environments to running the algorithms


## References
- The original GAIL paper: [link](https://arxiv.org/abs/1606.03476)
- Reinforcement Learning Collection with PyTorch: [link](https://github.com/hcnoh/rl-collection-pytorch)