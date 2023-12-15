from setuptools import find_packages, setup
from typing import List

HYPHON_E_DOT = '-e .'

def get_packages(file_path: str)->List[str]:

    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
    
    if HYPHON_E_DOT in requirements:
        requirements.remove(HYPHON_E_DOT)
    
    return requirements

setup(
    name='imagecaptioning',
    version = '0.0.1',
    author = 'Tushaar',
    author_email='tushaar.sharma05@gmail.com',
    packages=find_packages(),
    install_requires = get_packages('requirements.txt')
)