from setuptools import setup, find_packages

setup(
    name="ai-module",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "opencv-python",
        "numpy",
        "scikit-learn",
        "torch",
        "ultralytics"
    ],
    author="Chris Kevin",
    description="Real-time Face Recognition + PPE Detection System",
)