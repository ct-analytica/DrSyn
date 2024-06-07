from setuptools import setup, find_packages

setup(
    name='DrSyn',
    version='1.0.1',
    description='Drug synonym recognition tool',
    company='Precision Genetics',
    website='https://precisiongenetics.com',
    author='Clark Thurston',
    author_email='Clark.thurston@precisiongenetics.com',
    packages=find_packages(include=['DrSyn', 'DrSyn.*']),
    install_requires=[
        'pandas',
        'requests',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
