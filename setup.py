from setuptools import setup, find_packages

setup(
    name="eyewear_counter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python",
        "ultralytics",
        "numpy",
        "pandas",
        "aiohttp",
        "tqdm",
        "xlsxwriter"
    ],
    extras_require={
        "app": [
            "gradio",
            "nest_asyncio",
        ],
    },
    entry_points={
        "console_scripts": [
            "eyewear-counter-app=app.app:main",
        ],
    },
        package_data={
        '': ['weights/*', 'assets/*'], 
    },
    author="Ekaterina Solovyeva",
    description="Fast model for counting eyewear types in large sets of images.",
    python_requires=">=3.8",
)
