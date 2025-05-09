from setuptools import setup, find_packages

setup(
    name="eyewear_counter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python-headless",
        "ultralytics",
        "numpy",
        "aiohttp",
        "tqdm",
        "pandas",
        "gradio",         
        "nest_asyncio",
        "xlsxwriter"
    ],
    author="qksolov",
    description="Eyewear counter project with face detection and eyewear classification",
    python_requires=">=3.8",
)
