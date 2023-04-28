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
            'pytest==7.3.1',
            'imageai==3.0.3',
            'matplotlib==3.5.1',
            'mpi4py==3.1.4',
            'mpi4py_mpich==3.1.2',
            'mysql-connector==2.2.9',
            'mysqlclient==2.1.1',
            'PyMySQL==1.0.3',
            'numpy',
            'object_detection==0.0.3',
            'object_detection_0.1==0.1',
            'opencv_python==4.7.0.72',
            'pandas==1.5.3',
            'Pillow==9.5.0',
            'pycocotools==2.0.6',
            'pylabel==0.1.48',
            'scipy==1.8.0',
            'segment_anything==1.0',
            'scikit-image',
            'stochopy==2.2.0',
            'tensorflow==2.12.0'
        ]
)