from setuptools import find_packages, setup

with open('jamie/_meta.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='JAMIE',
    author='Noah Cohen Kalafut',
    description='Joint Autoencoders for Multi-Modal Imputation and Embedding',
    long_description=readme,
    long_description_content_type="text/markdown",
    version=__version__,
    packages=find_packages(exclude=('tests')),
    install_requires=[
        'adjustText',
        'anndata',
        'brokenaxes',
        'matplotlib',
        'numpy',
        'pandas',
        'pynndescent==0.5.8',  # UMAP Incompatibility
        'scanpy',
        'scikit-learn-extra',
        'scipy',
        'seaborn',
        'scikit-learn',
        'torch',
        'torchvision',
        'umap-learn',
        'unioncom',
    ],
    extras_require={
        'dev': [
            'pip-tools',
        ],
        'notebooks': [
            'mmd_wrapper @ git+https://git@github.com/Oafish1/WR2MD@v1.2.5a',
            'openpyxl',
            'shap',
        ],
    },
	tests_require=['pytest'],
)
