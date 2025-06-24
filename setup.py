from setuptools import setup, find_packages

setup(
    name="ObjectDetectionMetrics",
    version="0.1.0",
    author="Matheus Levy",
    author_email="matheus.levy@nca.ufma.br",
    description="Package for calculating object detection metrics",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/ObjectDetectionMetrics",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.1",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "pycocotools>=2.0.10",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
    keywords="object detection metrics coco precision recall map",
    license="MIT",
)