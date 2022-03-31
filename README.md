_DraCor Stilometry_

This repository contains the code, input data and example outputs for our DraCor-Project.

_Overview_

data_files				contains input data
expose,presentation,project_paper	contains our writeups
program_files				contains our code
visualization_output			contains our visualizations

_Running the code_

Clone this repository, run your code from this (main) directory as follows:

python3 program_files/visualization.py

This script is our main script. dracor\_data is responsible for requesting data from the DraCor API and featurizing the data, dracor\_nlp ist responsible for the lemmatization and POS-tagging. Those scripts will be called automatically by visualization.py.

_Requirements_

python3.6 or bigger
sklearn1.0.2
spacy, spacy-models it\_core\_news\_lg, de\_core\_news\_sm
nltk
internet connection while running the code
