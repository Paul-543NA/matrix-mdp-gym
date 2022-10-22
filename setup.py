import setuptools

_VERSION = "1.0.1"

short_description = "An OpenAI gym / Gymnasium environment to seamlessly create discrete MDPs from matrices."

REQUIRED_PACKAGES = [
    "gymnasium ~= 0.26.2",
    "numpy ~= 1.19.5",
]


# Loading the "long description" from the projects README file.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matrix-mdp-gym",
    version=_VERSION,
    author="Paul Festor",
    author_email="paul.festor@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Paul-543NA/matrix-mdp-gym",
    # Contained modules and scripts:
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # PyPI package information:
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT License",
    python_requires=">=3.6",
    keywords=' '.join([
        "Reinforcement-Learning",
        "Reinforcement-Learning-Environment",
        "Gym-Environment",
        "Markov-Decision-Processes",
        "Gym",
        "OpenAI-Gym",
        "Gymnasium",
    ]),
)
