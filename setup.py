from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ObjectDetectionMetrics",
    version="0.1.0",
    author="Matheus Levy",
    author_email="matheus.levy@nca.ufma.br",
    description="Package for calculating object detection metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/viplabufma/ObjectDetectionMetricsLibrary",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.1",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "pycocotools>=2.0.10",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research"
    ],
    python_requires='>=3.10',
    keywords="object detection metrics coco precision recall map",
    license="MIT",
    extras_require={
        'test': [
            'pytest>=8.2.2',
            'pytest-cov>=4.0',
            'coverage>=7.0'
        ],
        'dev': [
            'twine>=5.0.0',
            'build>=1.2.0',
            'ipython>=8.0.0'
        ]
    }
)