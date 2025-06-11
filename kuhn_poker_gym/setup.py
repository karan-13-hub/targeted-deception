from setuptools import setup, find_packages

setup(name='gym_kuhn_poker',
      version='0.1',
      description='OpenAI gym environment for Kuhn poker',
      packages=find_packages(),
      install_requires=['gym', 'numpy']
      )