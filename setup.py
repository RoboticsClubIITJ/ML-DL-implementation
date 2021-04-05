import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# TODO: Google Group for the team. And add that googlegroup id in author_email
setuptools.setup(
    name="MLlib",
    version="1.0.1",
    author="MLlib Development Team",
    author_email="singh.77@iitj.ac.in",
    description="""
    Package for ML and DL algorithms using nothing but numpy and matplotlib.
    """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RoboticsClubIITJ/ML-DL-implementation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Engineering",
        "Topic :: Engineering :: Machine Learning",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.0',
        'matplotlib>=3.0.0',
        'scipy'
        'pandas'
    ],
)
