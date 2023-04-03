from setuptools import setup

setup(
    name='hict',
    version='0.1.1rc1',
    packages=['hict', 'hict.api', 'hict.core', 'hict.util'],
    url='https://genome.ifmo.ru',
    license='',
    author='Alexander Serdiukov, Anton Zamyatin and CT Lab ITMO University team',
    author_email='',
    description='HiCT is a model for efficient interaction with Hi-C contact matrices that actively uses Split-Merge tree structures.',
    setup_requires=[
        'setuptools~=63.2.0',
        'wheel~=0.37.1',
    ],
    install_requires=[
        'h5py >= 3.7.0, < 3.8',
        'matplotlib >= 3.5.2, <3.7',
        'recordclass >= 0.17.2, < 0.19',
        'frozendict>=2.3.4, < 2.4',
        'scipy >=1.8.1, <1.10',
        'numpy >= 1.23.2, < 1.25',
        'cachetools>=5.2.0, <5.3',
        'bio>=1.3.9, < 1.6',
        'readerwriterlock>=1.0.9, < 1.1',
        'setuptools>=63.2.0, < 66',
        'wheel>=0.37.1, < 0.39',
    ],
    tests_require=[
        'cooler >=0.8.11, <0.9',
        'pytest >=7.2, <8',
        'pytest-quickcheck >=0.8.6, <1',
        'hypothesis >=6.61, <7',
        'mypy >=0.971, <1',
        'types-cachetools >=5.2.0, <6 ',
    ],
    test_suite='tests'
)
