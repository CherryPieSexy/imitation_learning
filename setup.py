from setuptools import setup

setup(
    name='cherry_rl',
    version='0.1.0',
    description='My awesome RL & IL library.',
    url='https://github.com/CherryPieSexy/imitation_learning',
    packages=['cherry_rl'],
    install_requires=['gym', 'numpy', 'scipy', 'six', 'torch', 'tensorboard', 'tqdm'],
)
