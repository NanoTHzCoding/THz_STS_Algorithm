from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='thz-sts-algorithm',
    version='0.0.1',
    description='',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NanoTHzCoding/THz-STS-Algorithm",
    author="Stefanie Adams",
    author_email="nanothz.coding@gmail.com",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn"
    ],
)
     