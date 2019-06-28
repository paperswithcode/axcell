import os
import re

import setuptools

directory = os.path.dirname(os.path.abspath(__file__))

# Extract version information
# path = os.path.join(directory, 'sota_extractor2', '__init__.py')
# with open(path) as read_file:
#     text = read_file.read()
# pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.MULTILINE)
# version = pattern.search(text).group(1)
version="2.0-alpha"

# # Extract long_description
# path = os.path.join(directory, 'README.md')
# with open(path) as read_file:
#     long_description = read_file.read()
long_description = ""
setuptools.setup(
    name='sota_extractor2',
    version=version,
    url='https://...',
    description='System for extracting data from arxiv papers',
    long_description_content_type='text/markdown',
    long_description=long_description,
    license='???',
    packages=setuptools.find_packages(),
    include_package_data=True,

    keywords='machine-learning ai information-extraction weak-supervision',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],

    project_urls={  # Optional
        'Homepage': 'https://...',
        'Source': 'https://...',
        'Bug Reports': 'https://...',
        'Citation': 'https://...',
    },
)