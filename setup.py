import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MLlib",
    version="0.0.0",
    author="MLlib Development Team",
    author_email="_@_.com",
    description="Package for ML and DL algorithms using nothing but numpy and matplotlib.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RoboticsClubIITJ/ML-DL-implementation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Engineering",
        "Topic :: Engineering :: Machine Leatning",
    ],
    python_requires='>=3.6',
)
