# -*- coding: utf-8 -*-

"""
This program assigns Parts-of-Speech tags to words by first training on the 
already tagged training text (de-train.tt) and then using viterbi algorithm to assign 
tags to the test text (de-test.t).

"""

import numpy as np
import nltk
from nltk.corpus.reader.conll import ConllCorpusReader
from collections import defaultdict
import timeit

# Reading from the corpus
root = "../data/"
ccorpus = ConllCorpusReader(root, ".tt", ('words', 'pos', 'tree'))


# Starting the timer for training
tag_train_start=timeit.default_timer()
"""
**************************************************************************************
**************************************************************************************

      CALCULATING INITIAL PROBABILITY
      
**************************************************************************************
**************************************************************************************
    
"""
"""
Retrieving the text as a list of list of sentences containing (word,tag) 
as tuples. Since every sentence is a new list it would be easier to retieve 
the first tag for initial probability 
"""
tagged_sent = ccorpus.tagged_sents("de-train.tt")

# Retrieving the first words along with their tags from the file to calculate initial probabilities
sent_start_list = [x[0] for x in tagged_sent]


initial_count=defaultdict(float)
initial_prob=defaultdict(float)

# this is the list of all tags that appear at the start of a sentence
list_first_tag=[x[1] for x in sent_start_list]


# A function to count the occurrence of different elements (word/tags) in the text
def element_counter(element_list): 
    element_count=defaultdict(float)
    for i in range(len(element_list)):
        if element_list[i] in element_count:
            element_count[element_list[i]]+=1
        else:
            element_count[element_list[i]]=1
    return(element_count)    

initial_count=element_counter(list_first_tag) # This list contains the count of each sentence beginner tag
        
        
# The initial probability (occurrence of tag1 as 1st tag of a sentence divided by total number of tags for the 1st word of all sentences)
for key in initial_count:
    initial_prob[key]=initial_count[key]/sum(initial_count.values())


"""
**************************************************************************************
**************************************************************************************

      CALCULATING TRANSITION PROBABILITY
      
**************************************************************************************
**************************************************************************************
    
"""

"""
Retrieving the text as a list of (word,tag) tuples. This format is more suitable 
for evaluating the bigrams of tags
"""
tagged_word = ccorpus.tagged_words("de-train.tt")
# extracting the tags from the (word,tag) tuples
list_of_tag=[x[1] for x in tagged_word]


# creating bigrams of tag pairs to calculate transition probability
bigram=[]
for i in range(len(list_of_tag)-1):
    x=list_of_tag[i]
    y=list_of_tag[i+1]
    bigram.append((x,y))

transition_count=element_counter(bigram)
tag_count=element_counter(list_of_tag)# Counting the total number of occurrence of each tag

# A function that calculates probability
def probability_calculator(item_count,tag_count):
    item_prob=defaultdict(float)
    second_item = [second for first, second in item_count] # Fetching the 2nd item in the dictionary key, which is a tuple
    
    j=0        
    for key in item_count:
        for i in tag_count:
            if second_item[j]==i:
                item_prob[key]=item_count[key]/tag_count[i] # dividing the total number of item pairs (item1,item2) by the total number of item2 in the text
        j+=1
    return(item_prob)

"""
Calculating the final transition probabilities
"""
transition_prob=probability_calculator(transition_count,tag_count)

  
"""
**************************************************************************************
**************************************************************************************

      CALCULATING EMISSION PROBABILITY
      
**************************************************************************************
**************************************************************************************
    
"""   
emission_count=element_counter(tagged_word)
emission_prob=probability_calculator(emission_count,tag_count)

"""
**************************************************************************************
**************************************************************************************

      CALCULATING VITERBI
      
**************************************************************************************
**************************************************************************************
   
""" 

# Creating a dictionary of unique words parsed in the training corpus
list_of_word=[x[0] for x in tagged_word]
dict_list_of_word=element_counter(list_of_word)

states_tags=sorted(initial_prob.items())
tag_list=[x[0] for x in states_tags]
tag_array=np.array(tag_list) #tag_array contains all the unique tags in alphabetical order

tag_train_end=timeit.default_timer() # Training of program ends here

# From here we start to test our program on the test data 
tag_test_start=timeit.default_timer()


"""
Reading from the test file
"""


with open("de-test.t", "r", encoding="utf-8") as test_file:
    with open("tagged_output.tt", "w", encoding="utf-8") as file_output: # tagged_output is the file that will have the test words and their tags
        print(test_file)
        test_data=[]
        new_line=[]
        for line in test_file:
            line=line.strip("\n")
            if line !='':
                new_line.append(line)
            else:
                test_data.append(new_line)
                new_line=[]
                
        observation_seq=[]
        for sentence in test_data:
            observation_seq=sentence
            
            obs_size=len(observation_seq)
            no_states=12
            path=np.zeros((obs_size,no_states))# the array that would store the states/tags corresponding to each viterbi value
            temp=np.zeros((obs_size,no_states))# a temporary array to store the intermediate viterbi calculations
            v=np.zeros((obs_size,no_states))# this is to be the Viterbi matrix
            
            # Counters that would assist in creating the Viterbi matrix
            i1=0
            j1=0
            k1=0
            
            for i in observation_seq:
                j1=0
                
                
                for j in tag_list:
        
                    
                    #Checking whether the word is unseen or not, accordingly performing smoothing
                    if (str(i),j) not in emission_prob: 
                        if str(i) not in dict_list_of_word:# dict_list_of_word contains all unique words of the training text 
                            emission_prob[(str(i),j)]=1
                        
                    
                    if (str(i),j) in emission_prob:
                    
                        
                        
                        if i1 == 0:
                            v[i1][j1]=initial_prob[j]*emission_prob[(str(i),j)]
                            path[i1][j1]=j1
                            
                        else:
                        
                            k1=0
                            max=0
                            max_path=0
                            for k in tag_list:
                                if (j,k) in transition_prob: 
                           
                                    temp[i1][k1]=emission_prob[(str(i),j)]*transition_prob[(j,k)]*v[i1-1][k1]
                                    
                                    
                                if temp[i1][k1]>max:
                                    max=temp[i1][k1]
                                    max_path=k1
                                v[i1][j1]=max
                                path[i1-1][j1]=max_path
                                k1+=1
                                
                                
                    j1+=1
                i1+=1

           
            
            
            
            # Checking for the backpointer path and storing it in next_path[]
            max_final=0
            for k in range(no_states):
                if v[obs_size-1][k]>max_final:
                    max_final=v[obs_size-1][k]
                    path_final=k
            next_path=np.zeros((obs_size))# this array would store the final set of tags
            next_path[obs_size-1]=path_final
      
            for x in range(obs_size-2,-1,-1):
                next_path[x]=path[x][int(next_path[x+1])]
                
            """
            # *************************************************************************************************                
            # ALTERNATE METHOD OF USING VITERBI WITH GREEDY METHOD TO ASSIGN TAGS INSTEAD OF USING BACKPOINTERS                
                
           
                
            max_final=0
            next_path=np.zeros((obs_size))
            for i in range(obs_size):
               max_final=0
               for k in range(no_states):
                   if v[i][k]>max_final:
                       max_final=v[i][k]
                       next_path[i]=k
                       
                       
            for i in range(obs_size):
                for j in range(len(tag_array)):
                    if next_path[i]==j:
            *************************************************************************************************                        
                        
            """
            
            
            
            """
            Obtaining the tags relative to the indices stored in next_path[]
            and consequently writing the word-tag mapping onto the output file
            """
            for i in range(obs_size):
                for j in range(len(tag_array)):
                        if next_path[i]==j:
                            print(f'{str(observation_seq[i])}\t{str(tag_array[j])}', file=file_output)
                            
            print(''.join("\n"),end="", file=file_output)                
       
tag_test_end=timeit.default_timer()
print("Total time to train: ",tag_train_end-tag_train_start)
print("Total time to tag using Viterbi: ",tag_test_end-tag_test_start)


