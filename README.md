# flower classification
## this repo has been deprecated - it was built in Lua Torch, which is mostly obsolete
NOTE:
Current model runs on the default 17 class subset of original dataset

To setup enviornment:

make sure you have virtual env and virtual env wrapper; if not, type:

```bash
pip install virtualenv virtualenvwrapper
```

and then put these two lines in your `.bashrc` or `.bash_profile`:

```bash
# Virtualenv/VirtualenvWrapper
source /usr/local/bin/virtualenvwrapper.sh
```

clone the repo in your working directory and type:

```bash
. setup.sh
```

NOTE: if python starts to make your terminal session act strangely, type this command in terminal (assuming you have macports installed):
```bash
sudo port selfupdate && sudo port clean python27 && sudo port install python27 +readline
```

In order to run trian-val-test with default config, type in

```bash
qlua main.lua
```
