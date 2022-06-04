from setuptools import setup

setup(
    name='hitree',
    version='1.0rc1.dev1',
    packages=['hitree', 'hitree.api', 'hitree.core', 'hitree.util', 'hitree.util.cool2chunks'],
    url='https://genome.ifmo.ru',
    license='',
    author='Alexander Serdiukov, Anton Zamyatin, Nikita Alexeev and CT Lab ITMO University team',
    author_email='',
    description='Hi-Tree is a model for efficient interaction with Hi-C contact matrices that actively uses Split-Merge tree structures.',
    install_requires=[
        'h5py~=3.6.0',
        'recordclass~=0.16.3',
        'frozendict~=2.3.0',
        'scipy~=1.8.0',
        'numpy~=1.21.3',
        'cachetools~=5.1.0',
        'bio~=1.3.8',
        'readerwriterlock~=1.0.9'
    ],
)
