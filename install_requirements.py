import pkg_resources,os

with open('GUI'+os.sep+'requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

for requirement in requirements:
    try:
        pkg_resources.require(requirement)
    except pkg_resources.DistributionNotFound:
        print(f"Could not find package: {requirement}")
    except pkg_resources.VersionConflict:
        print(f"Version conflict for package: {requirement}")
