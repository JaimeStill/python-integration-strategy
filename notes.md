# Python Notes

## Set Virtual Environment in VS Code

1. <kbd>F1</kbd>  > **Python: Select Interpreter**

2. Select **Enter interpreter path...**

3. Select `.venv\Scripts\python.exe`

## VS Code Settings

The following Python extension settings are helpful:

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.inlayHints.functionReturnTypes": true,
    "python.analysis.inlayHints.variableTypes": true
}
```

## Pip Commands

```bash
# show installed packages along with install location
py -m pip list -v

# list available versions of a package
py -m pip index versions [package]

# show package information
# includes install location
py -m pip show [package]

# download dependencies tree from requirements.txt
py -m pip download -r requirements.txt -d [output-directory]

# archive dependencies
py -m tar cvfz [output-directory].tar.gz [output-directory]

# unpack dependencies
py -m tar zxvf [output-directory].tar.gz

# install dependencies
cd [output-directory]
# linux
py -m pip install * -f ./ --no-index
# windows
for %f in (*.whl) do pip install --no-index --find-links=./ %f

# uninstall packages
py -m pip uninstall [packages] -y

# create a virtual environment
py -m venv [path-to-venv]

# example virtual environment
py -m venv .\.venv

# activate virtual environment
.venv\Scripts\Activate.ps1

# deactivate virtual environment
deactivate

# update pip
py -m pip install --upgrade pip
```