# Read the installed packages
with open('freeze.txt', 'r') as f:
    freeze_packages = {line.strip().lower() for line in f}

# Read the packages in requirements.txt
with open('requirements.txt', 'r') as f:
    requirements_packages = {line.split('==')[0].strip() for line in f}

# Get the intersection
intersection_packages = {pkg for pkg in freeze_packages if pkg.split('==')[0] in requirements_packages}

# Print the intersection
for package in intersection_packages:
    print(package)