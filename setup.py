from setuptools import setup, find_packages

setup(
    name="FastThinkNet",
    version="0.1.0",
    author="VishwamAI",
    author_email="kasinadhsarma@gmail.com",
    description="A neural network library for fast thinking agents, integrating PyTorch and TensorFlow.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/FastThinkNet",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
