from setuptools import setup

setup(
    name='torch_rl',
    version='0.1.0',
    description='My awesome RL & IL library.',
    url='https://github.com/CherryPieSexy/imitation_learning',
    packages=['torch_rl'],
    install_requires=['gym', 'numpy', 'scipy', 'six', 'torch', 'tensorboard', 'tqdm'],
)
