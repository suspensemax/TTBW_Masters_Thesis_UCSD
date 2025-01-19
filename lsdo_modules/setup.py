from setuptools import setup, find_packages

setup(
    name='lsdo_modules',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
        'myst-nb',
        'sphinx_rtd_theme',
        'sphinx-collections',
#         'scipy',
#         'pint',
#         'sphinx-rtd-theme',
#         'sphinx-code-include',
#         'jupyter-sphinx',
#         'numpydoc',
    ],
)
