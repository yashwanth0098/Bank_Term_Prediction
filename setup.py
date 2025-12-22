from setuptools import setup, find_packages
from typing import List 

def get_requirements(file_path: str) -> List[str]:
    """Read the requirements from a file and return them as a list."""
    try:
        requirements_lst:List[str] = []
        with open("requirements.txt", 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement !="-e .":
                    requirements_lst.append(requirement)
    except FileNotFoundError:
        print("Requirements file not found.")
    return requirements_lst

setup(
    name="my_package",
    version="0.1.0",
    author="Always_my name",
    author_email="Yashwanth00998gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)