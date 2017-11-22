from setuptools import find_packages
from setuptools import setup
setup(
    name='trainer',
    version='2.1',
    packages=find_packages(),
    install_requires=[
        'pandas == 0.20.3',
        'scikit-learn >= 0.19.1',
        'keras',
        'h5py'
    ],
    zip_safe=False
)

