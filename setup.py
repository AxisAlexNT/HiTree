from typing import List
from setuptools import find_packages, setup


requirements: List[str] = []
with open("requirements.txt", mode="rt", encoding="utf-8") as f:
    requirements = f.readlines()

setup(
    name='hict',
    version='0.1.1rc1',
    packages=list(set(['hict', 'hict.api', 'hict.core', 'hict.util']).union(find_packages())),
    url='https://genome.ifmo.ru',
    license='',
    author='Alexander Serdiukov, Anton Zamyatin and CT Lab ITMO University team',
    author_email='',
    description='HiCT is a model for efficient interaction with Hi-C contact matrices that actively uses Split-Merge tree structures.',
    setup_requires=[
        'setuptools>=63.2.0',
        'wheel>=0.37.1',
    ],
    install_requires=list(set([]).union(requirements)),
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
