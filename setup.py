from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return list(set(lines))

setup(
    name='ai_mental_health_coach',
    version='0.1.0',
    description='An AI-powered mental health coaching application.',
    author='Back Propagators',
    author_email='developer@aydie.in',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
