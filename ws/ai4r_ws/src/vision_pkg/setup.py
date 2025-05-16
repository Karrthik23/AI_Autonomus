from setuptools import setup

package_name = 'vision_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
    ('share/' + package_name, ['package.xml']),
    ('share/ament_index/resource_index/packages',
     ['resource/' + package_name]),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='YOLOv8 + DeepSORT vision node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_deepsort_node = vision_pkg.yolo_deepsort_node:main',
            'webcam_publisher = vision_pkg.webcam_publisher:main',
        ],
    },
    
)
