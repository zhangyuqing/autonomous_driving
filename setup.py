from distutils.core import setup

def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()

setup(
    name='detra',
    version='0.1',
    description='AI system of object detection and tracking',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/zhangyuqing/autonomous_driving',
    author='Yuqing Zhang',  
    author_email='yqng.zh@gmail.com',  
    license='MIT',
    packages=['detra'],
)