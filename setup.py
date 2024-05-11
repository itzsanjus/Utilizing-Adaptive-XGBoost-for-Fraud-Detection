from setuptools import find_packages,setup

HYPHEN_E_DOT = '-e .'

def get_packages(file_path:str)->list:
    '''
        Input: Text file of requirements
        Output: List of requirements
    '''
    requirements = []
    with open (file_path) as package_list:
        requirements = package_list.readlines()
    requirements = [req.replace('\n','') for req in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name= 'Fraud_Detection_XGBoost', 
    author= 'Sanju Sarkar', 
    author_email= 'sanjusarkar44@hotmail.com',
    version = '0.0.1',
    packages=find_packages(),
    install_requires = get_packages('requirements.txt')
    )