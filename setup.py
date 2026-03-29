from setuptools import find_packages,setup
from typing import List

hyphen_e_dot = "-e ."
def get_requirement(path:str) -> List[str]: #return list of requirements
    requirements = []
    with open(path) as file:
        requirements = file.readlines()
        requirements= [req.replace("\n","") for req in requirements]
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
        return requirements
        

setup(
    name = "MLProject",
    version= "0.0.1",
    author= "Devansh Jaiswal",
    author_email="devj59@gmail.com",
    packages=find_packages(),
    install_requires=get_requirement("requirements.txt")
)