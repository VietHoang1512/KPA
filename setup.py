import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="qs-kpa",
    version="0.0.1",
    description="Quantitative Summarization – Key Point Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VietHoang1512/KPA",
    author="Phan Viet Hoang",
    author_email="phanviethoang1512@gmail.com",
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Natural Language :: English",
    ],
    python_requires=">=3.7",
    packages=setuptools.find_packages(exclude=["scripts", "bin", ".circleci", "assets", "paper"]),
    package_dir={"qs_kpa": "qs_kpa"},
    include_package_data=True,
    install_requires=[
        "transformers>=4.1.0,<5.0.0",
        "tqdm",
        "torch>=1.6.0",
        "numpy",
        "sklearn",
        "scipy",
        "pandas",
        "gdown",
        "pytorch-metric-learning",
    ],
)
