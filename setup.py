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
        'h5py~=3.7.0',
        'matplotlib~=3.5.2',
        'recordclass~=0.17.2',
        'frozendict~=2.3.4',
        'scipy~=1.8.1',
        'numpy~=1.23.1',
        'cachetools~=5.2.0',
        'bio~=1.3.9',
        'readerwriterlock~=1.0.9',
    ],
    tests_require=[
        'pytest~=5.4.3',
        'pytest-quickcheck~=0.8.6'
    ],
    test_suite='tests'
)
