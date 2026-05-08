import setuptools

VERSION = '1.1.0'

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DMC",
    version=VERSION,
    description="AT-Dec-POS Doudizhu research stack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    keywords=["DouDizhu", "AI", "Reinforcment Learning", "RL", "Torch", "Poker"],
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'rlcard',
        'matplotlib',
    ],
    requires_python='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
