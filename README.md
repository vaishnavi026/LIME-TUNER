# LIME-TUNER
##### The LIME_TUNER_LIBRARY contains :
* the modified code of LIME (the original code was taken from the github, I have made the changes  in it)
* tuner_library.py - class Tuner is defined here, which have arguments as the text instance for which you need the explanation, the filename (blackbox model in the form of serialize .pkl file), and the class_names(the names of target classes, which will be used by LIME code)
##### Note : This code works for binary text classification dataset.
* utilities.py - Contains code for the functions, which are used by Lime_Tuner_Workflow.
* Lime_Tuner_Workflow : demostrates the workflow using lime_tuner library
* model.pkl : sample serialized bb model
* explanation_25.html : sample of the explanation

###### requiremts.txt > versions required to run the code
