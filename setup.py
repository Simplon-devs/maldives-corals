"""
IMPORTANT : the list of dependencies is note definitive and will need
to be reviewed (some dependencies might be unneeded).

The following dependencies might be required:
- sudo apt-get install libmysqlclient-dev
- sudo apt-get install libopenmpi-dev
"""

from distutils.core import setup

setup(name='maldives_corals',
        version='0.1.0',
        description='Analyzing coral restoration with machine learning',
        author='G. Morand, Simplon.co',
        url='https://github.com/Simplon-devs/maldives-corals',
        packages=['maldives_corals'],
        install_requires=[
            'pytest',
            'imageai',
            'numpy',
            'opencv_python',
            'Pillow',
            'pylabel',
            'torch',
            'torchvision',
            'torchaudio'
        ]
)