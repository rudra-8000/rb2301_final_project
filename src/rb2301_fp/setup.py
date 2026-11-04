from setuptools import find_packages, setup
from glob import glob

package_name = 'rb2301_fp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['optitrack_variables.config']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'numpy', 'matplotlib'],
    zip_safe=True,
    maintainer='marmot',
    maintainer_email='marmot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_course = rb2301_fp.obstacle_course:main',
            'visualization_relay = rb2301_fp.visualization_bridge:main',
        ],
    },
)
