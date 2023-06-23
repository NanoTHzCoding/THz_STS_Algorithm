import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='thzsts',
    version='0.0.3',
    author="Stefanie Adams",
    author_email="nanothz.coding@gmail.com",
    description='Function to perform THz STS algorithm and simulate measurements.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ], 
    py_modules=['thzsts'],
    package_dir={'': 'thzsts/src'},
    url="https://github.com/NanoTHzCoding/THz_STS_Algorithm",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn"
    ],
)
     