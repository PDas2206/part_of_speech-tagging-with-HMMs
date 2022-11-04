# part-of-speech tagging with HMMs
We implement a bigram part-of-speech (POS) tagger based on Hidden Markov Models (without using NLTK except for the modules explicitly listed below). The project is divided into the following modules:
1. Corpus reader and writer
2. Training procedure, including smoothing
3. Viterbi tagging, including unknown word handling
4. Evaluation
## Viterbi algorithm
We implement the Viterbi algorithm for finding the optimal state (tag) sequence given the sequence of observations (words). One can test the implementation on a small example for which you know the correct tag sequence, such as the Eisner's Ice Cream HMM. 

## Training
The program then learns the parameters of the HMM from the data: the initial, transition, and emission probabilities. A maximum likelihood training procedure for supervised learning of HMMs is then implemented. The corpus has been sourced from http://www.coli.uni-saarland.de/~koller/materials/anlp/de-utb.zip. It contains a training set, a test set, and an evaluation set. The training set (de-train.tt) and the evaluation set (de-eval.tt) are written in the commonly used CoNLL format. They are text files with two colums; the first column contains the words, the POS tags are in the second column, and empty lines delimit sentences. The corpus uses the 12-tag universal POS tagset by Petrov et al. (2012). 
## Evaluation
The trained model is evaluated on the unseen data from the test set. The Viterbi algorithm is executed over each of the models, outputing a tagged corpus in the two-column CoNLL format (*.tt). 

## Getting Started:
The program in this repo have been done using Python 3. To run the files one has to have Python 3 or  above installed in their systems. Additionally you need the following packages to be installed: \
		nltk		To read the corpus \
		numpy		To create and work with numpy arrays \
		collections	To work with defaultdict \
		timeit		To record the time needed for the program to perform its various tasks \

The statements required to import these are included in the programs, and so one does'nt have to explicitly write them while executing the program.

## Pre-requisite:
	You need to have a folder named "data" in which you need to store this program, the training text (de-train.tt), test text (de-test.tt) and the evaluating
	program eval.py. 

## Running the program:
	To execute the programs from the command shell, do the following steps: \
		1. Change the directory to this folder (named "data") in your system. \
		2. To execute the program and generate the .tt file containing the words of the test corpus along with their respective tags,
		   type the following after the prompt appears: \
			python pos.py \
			This will generate the file named tagged_output.tt that has words and their tags in CoNLL format. Along with that it would also 
			display the time taken to train and to tag the test data. \
		3. To evaluate this generated file you need to have the gold standard file for POS tagging, de-eval.tt, and the program that 
		   would evaluate the accuracy of tagged_output with respect to the gold standard file, namely eval.py, in this very same folder. \
		4. To now evaluate this generated file, type the following in the command prompt (considering you are still in the directory of this folder):
			python eval.py de-eval.tt tagged_output.tt \
      (considering you have access to eval.py or any other such evaluation script for part-of-speech tagging)
		5. The corresponding accuracy measure will now be displayed on to the screen.
