from setuptools import setup, find_packages

main_module = 'sidetopics'
project_name = 'sidetopics'
__author__ = 'Bryan Feeney'
__author_email__ = 'bryan.feeney@gmail.com'


VERSION = '0.1.0'

setup(
    name=project_name,
    author=__author__,
    author_email=__author_email__,
    python_requires='>=3.6',
    version=VERSION,
    packages=find_packages(),
    # namespace_packages=namespace_packages,
    entry_points={
        'console_scripts': [
            f'{project_name} = {main_module}.__main__:main',
        ]
    },
    package_data={
        f'{main_module}': ['versionfile']
    }
)
