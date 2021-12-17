# AI lectures:

* Introduction to Artificial Intelligence
* Introduction to Robotics
* Introduction to Machine Learning
* Maths for Intelligent Systems

## Introduction to Machine Learning

* [Full script](../../releases/download/v0.1/script.pdf) from summer 2019
* [Individual slides and exercises with solutions](../../releases/tag/v0.1)

## Introduction to Optimization

* [Full script](../../releases/download/v0.2/script.pdf) from summer 2015
* [Individual slides and exercises with solutions](../../releases/tag/v0.2)

## Compilation

Ubuntu packages to be installed:
```
sudo apt-get install \
	texlive-latex-recommended \
	texlive-latex-extra \
	texlive-science \
	fig2dev #or transfig for Ubuntu 16.04 !
```

Generate the shared pdf pics (from fig-files)
```
cd shared/pics
./compile.sh
```

Compile a single lecture
```
cd MachineLearning/pics
./compile.sh
```

Compile a single lecture
```
cd MachineLearning
pdflatex -interaction=nonstopmode 01-introduction
pdflatex -interaction=nonstopmode 01-introduction
```

Compile a whole set of lectures
```
cd MachineLearning
./compile.sh
```
