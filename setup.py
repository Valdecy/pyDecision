from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pyDecision',
    version='4.5.8',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyDecisions',
    packages=find_packages(),
    install_requires=[
        'llmx',
        'matplotlib',
        'numpy',
        'openai',
        'pandas',
        'scikit-learn',
        'scipy'
    ],
    description='A MCDA Library Incorporating a Large Language Model to Enhance Decision Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
