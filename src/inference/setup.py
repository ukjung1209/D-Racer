import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='topst',
    maintainer_email='sooyong.park@telechips.com',
    description='Autonomous inference nodes: lane detection, object detection, decision making.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_node = inference.lane_node:main',
            'object_node = inference.object_node:main',
            'decision_node = inference.decision_node:main',
            'decision_arrow_node = inference.decision_arrow_node:main',
            'decision_light_node = inference.decision_light_node:main',
            'decision_obstacle_node = inference.decision_obstacle_node:main',
        ],
    },
)
