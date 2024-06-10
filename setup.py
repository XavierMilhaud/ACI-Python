# setup.py
# A modifier 

from setuptools import setup, find_packages

setup(
    name="mon_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Ajoutez ici les dépendances de votre package
    ],
    author="Kanee",
    author_email="votre.email@example.com",
    description="Une description de votre package",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/votre-utilisateur/mon_package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

