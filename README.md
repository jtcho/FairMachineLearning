# Rawlsian Fair Machine Learning for Contextual Bandits

Implementation and evaluation of provably Rawlsian fair ML algorithms for contextual bandits.

Related Work/Citations:

* Rawlsian Fairness for Machine Learning (https://arxiv.org/abs/1610.09559)
* Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms (https://arxiv.org/abs/1003.5956)

## Installation Instructions

### (Option 1) Setting Up virtualenv

#### OSX

Install Python 3 from [package](https://www.python.org/downloads/). This allows you to run `python3` and `pip3`. Software is installed into `/Library/Frameworks/Python.framework/Versions/3.x/bin/`.

Install virtualenv for Python 3 for the user only (which is placed into `~/Library/Python/3.x/bin`):

```
$ pip3 install --user virtualenv
```

Create the following alias in your `~/.bash_profile`:

```
$ echo "alias virtualenv3='~/Library/Python/3.x/bin/virtualenv'" >> ~/.bash_profile
```

Create a local virtualenv and activate it:

```
$ virtualenv3 fairml
$ source fairml/bin/activate
```

With the virtualenv active, install the project requirements into your virtualenv:

```
$ pip install -r requirements.txt
```

Create a Python kernel for Jupyter that uses your virtualenv:

```
$ python -m ipykernel install --user --name=fairml
```

You can then launch Jupyter using `jupyter notebook` from inside the project directory and change the kernel to `fairml`.

### (Option 2) Using Docker

You can install [Docker](https://www.docker.com) and use a standard configuration such as `all-spark-notebook` to run the project files.
