from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pydecision',
    version='5.1.1',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyDecisions',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'pyDecision.web': [
            'templates/*.html',
            'static/css/*.css',
            'static/js/*.js',
            'static/img/*',
            'examples.json',
        ],
    },
    install_requires=[
        'flask>=2.0',
        'google-genai',
        'llmx',
        'matplotlib',
        'networkx',
        'numpy',
        'openai',
        'pandas',
        'scikit-learn',
        'scipy',
        'werkzeug>=2.0',
    ],
    description='A MCDA Library Incorporating a Large Language Model to Enhance Decision Analysis. Now with a built-in Flask web GUI launchable via pyDecision.web_app().',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
