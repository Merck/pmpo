from setuptools import setup, find_packages
import re
import ast

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('pMPO/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(
    name='pMPO',
    version=version,
    packages=find_packages(exclude=['*tests*', '*test*']),
    url='http://www.merck.com',
    license='Apache 2.0',
    author='Scott Arne Johnson',
    author_email='scott.johnson6@merck.com',
    description='Probabilistic MPO models',
    install_requires=['numpy>=1.11.3', 'scipy>=0.18.1', 'pandas>=0.19.2', 'statsmodels>=0.6.1'],
    test_suite='pMPO.tests',
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
    ]
)
