# A Notebook about NLP Spacy course

## About this notebook

This notebook is based on Ines Montani's free online course, available at:

https://course.spacy.io/

https://github.com/ines/spacy-course

## How to install spacy:

The instructions are in the link below :

https://spacy.io/usage

To install spacy, please use:

<pre><code>conda install -c conda-forge spacy</code></pre>

This notebook is tested for version 2.1.3

As per May, 4th 2019, installing Spacy via "conda install -c conda-forge spacy" delivers 2.0.8 version so I used "pip install -U spacy" to have version 2.1.3 in my computer

To check the library version, use the commands below:

<pre><code>import spacy
print(spacy.__version__)
</code></pre>


## How to download the models:

https://github.com/explosion/spacy-models/releases/

To download a model:

<pre><code>python -m spacy download pt
python -m spacy download en
</code></pre>

or...
<pre><code>python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm
python -m spacy download en_core_web_md
</code></pre>

To link a model to refer to it more easily:

<pre><code>python -m spacy link en_core_web_md en
python -m spacy link pt_core_news_sm pt
</code></pre>


## Using nbextensions

I highly recommend you to install table of contents from nbextensions, that makes the navigation in the sections much more easier.

Instructions can be found here:

https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html

<pre><code>conda install -c conda-forge jupyter_contrib_nbextensions</code></pre>
<pre><code>jupyter nbextension enable toc2</code></pre>

## Additional information

If you have issues rendering these notebooks, you can try nbviewer

https://nbviewer.jupyter.org/github/Cristianasp/spacy/blob/master/Chapter_01.ipynb

https://nbviewer.jupyter.org/github/Cristianasp/spacy/blob/master/Chapter_02.ipynb

https://nbviewer.jupyter.org/github/Cristianasp/spacy/blob/master/Chapter_03.ipynb

https://nbviewer.jupyter.org/github/Cristianasp/spacy/blob/master/Chapter_04.ipynb

## Or you want to use binder ?

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cristianasp/spacy/master)


## LICENSE

The materials are licensed under CC BY-SA.

See here for details: [https://github.com/ines/spacy-course/blob/master/LICENSE]
