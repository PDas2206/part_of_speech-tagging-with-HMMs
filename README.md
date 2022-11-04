# part_of_speech-tagging-with-HMMs
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
