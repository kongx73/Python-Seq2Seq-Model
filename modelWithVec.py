# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 23:10:43 2017

@author: Procheta
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:45:05 2017

@author: Procheta
"""
import numpy as np
import random
from random import shuffle
import tensorflow as tf
import itertools


#code where word Id s aretaken not  the word vectors 

NUM_EXAMPLES = 100
NUM_TEST_EXAMPLES = 4
VOCABULARY_SIZE = 232
WORDS_PER_QUERY = 5
HIDDEN_LAYER_WORD = 10
HIDDEN_LAYER_QUERY = 10
HIDDEN_LAYER_DECODER = 10
NUM_QUERY_SESSION = 2
NUM_EPOCH_WORD = 5
NUM_EPOCH_QUERY = 5
NUM_EPOCH_SESSION= 5
NUM_EPOCH_NEXTWORD= 5

tf.set_random_seed(42)
tList = [0,0,0,0,0,0,0,0,0,0]
# load the training query  word Id file 
train_input  = []
with open("C:/Users/Procheta/Desktop/newQueryVecFile.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        words = line1[0].split(',')
        tempList=[]
        for word in words:
           dim = word.split(' ',11)
           dim1 = dim[1:11]
           x1 = np.array(dim1)
           y1 = x1.astype(np.float)
           tempList.append(y1)
        if len(words) < WORDS_PER_QUERY:
             for i in range(WORDS_PER_QUERY - len(words)):
                 x1 = np.array(tList)               
                 y1 = x1.astype(np.float)
                 tempList.append(y1)
        x = np.array(tempList)
        y = x.astype(np.float)
        train_input.append(y)
           
         
                 

train_input = train_input[:NUM_EXAMPLES]
# load the testing query  word Id file 
test_input=[]
with open("C:/Users/Procheta/Desktop/newQueryVecFile.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        words = line1[0].split(',')
        tempList=[]
        for word in words:
           dim = word.split(' ',11)
           dim1 = dim[1:11]
           x1 = np.array(dim1)
           y1 = x1.astype(np.float)
           tempList.append(y1)
        if len(words) < WORDS_PER_QUERY:
             for i in range(WORDS_PER_QUERY - len(words)):
                 x1 = np.array(tList)               
                 y1 = x1.astype(np.float)
                 tempList.append(y1)
        x = np.array(tempList)
        y = x.astype(np.float)
        test_input.append(y)
           
        
        

# load the test output data        
test_output=[]
count = 0
with open("C:/Users/Procheta/Desktop/data/finalTest.txt") as fp:
    for line in fp:
        line1 = line.split('\n')
        if count >= NUM_QUERY_SESSION :
            test_output.append(line1[0])
        count = count + 1
        
#load the target output for the word level LSTM
train_output = []
with open("C:/Users/Procheta/Desktop/newQueryVecFile.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        words = line1[0].split(',')
        tempList=[]
        count = 0
        for word in words:
           if count == len(words) - 1:
               dim = word.split(' ',11)
               dim1 = dim[1:11]
               x1 = np.array(dim1)
               y1 = x1.astype(np.float)
               tempList.append(y1)       
               x = np.array(tempList)
               y = x.astype(np.float)
               train_output.append([val for sublist in y for val in sublist])
               
           count = count + 1

print ("Test and Training data loaded for word level LSTM:")


# Designing the lstm in word level
data = tf.placeholder(tf.float32, [None, WORDS_PER_QUERY,10]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 10])
num_hidden = HIDDEN_LAYER_WORD
#cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.nn.rnn_cell.GRUCell(num_hidden)
val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)


# function to encode session

def newFunc(myList=[], test_input=[], targetOutput =[]):
    
 #model design   
 queryData = tf.placeholder(tf.float32, [None, NUM_QUERY_SESSION,HIDDEN_LAYER_WORD])
 queryTarget = tf.placeholder(tf.float32, [None, HIDDEN_LAYER_WORD])
 queryCell = tf.nn.rnn_cell.GRUCell(HIDDEN_LAYER_QUERY)
 val, state = tf.nn.dynamic_rnn(queryCell, queryData, dtype=tf.float32)
 val = tf.transpose(val, [1, 0, 2])
 queryLast = tf.gather(val, int(val.get_shape()[0]) - 1)
 queryWeight = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_QUERY, int(queryTarget.get_shape()[1])]))
 queryBias = tf.Variable(tf.constant(0.1, shape=[queryTarget.get_shape()[1]]))
 queryPrediction = tf.nn.softmax(tf.matmul(queryLast, queryWeight) + queryBias)
 query_cross_entropy = -tf.reduce_sum(queryTarget * tf.log(tf.clip_by_value(queryPrediction,1e-10,1.0)))
 query_optimizer = tf.train.AdamOptimizer()
 query_minimize = query_optimizer.minimize(query_cross_entropy) 


 #data preperation
 train_input=[]
 train_output=[]
 
 for i in range(len(myList) - NUM_QUERY_SESSION):
  xi=[]
  xi.append(list(itertools.chain.from_iterable(myList[i])))
  xi.append(list(itertools.chain.from_iterable(myList[i+1])))
  x = np.array(xi)
  y= x.astype(np.float)
  train_input.append(y) 
  
  
 for i in range(len(targetOutput)):
  train_output.append(list(itertools.chain.from_iterable(targetOutput[i])))
  
 test=[]
 for i in range(len(test_input) - NUM_QUERY_SESSION):
  xi=[]
  xi.append(list(itertools.chain.from_iterable(test_input[i])))
  xi.append(list(itertools.chain.from_iterable(test_input[i+1])))
  x = np.array(xi)
  y= x.astype(np.float)
  test.append(y) 
  
 # executing the model   
 init_op = tf.global_variables_initializer()
 sess = tf.Session()
 sess.run(init_op)

 batch_size = 1
 no_of_batches =int(int(len(train_input)) / batch_size)
 epoch = NUM_EPOCH_QUERY
 for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
      inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
      ptr+=batch_size
      sess.run(query_minimize,{queryData: inp, queryTarget: out})
    print ("Epoch For Session Encoding LSTM: ",str(i))
 outputData = []
 for i in range(len(train_input)):
     stateOutput = sess.run(state,{queryData: [train_input[i]]})
     outputData.append(stateOutput)     
 outputTestData = []
 for i in range(len(test)):
     testStateOutput = sess.run(state,{queryData: [test[i]]})
     outputTestData.append(testStateOutput)
 
 
 sess.close()

 return outputData,outputTestData;

#function to predict the first word of the predicted query

def decoderStage(myList=[], testInput=[]):
    
 # designing the model
 decoderData = tf.placeholder(tf.float32, [None, 1, HIDDEN_LAYER_QUERY])
 decoderTarget = tf.placeholder(tf.float32, [None, VOCABULARY_SIZE])
 decoderCell = tf.nn.rnn_cell.GRUCell(HIDDEN_LAYER_DECODER)
 val, state = tf.nn.dynamic_rnn(decoderCell, decoderData, dtype=tf.float32)
 val = tf.transpose(val, [1, 0, 2])
 decoderLast = tf.gather(val, int(val.get_shape()[0]) - 1)
 decoderWeight = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_DECODER, int(decoderTarget.get_shape()[1])]))
 decoderBias = tf.Variable(tf.constant(0.1, shape=[decoderTarget.get_shape()[1]]))
 decoderPrediction = tf.nn.softmax(tf.matmul(decoderLast, decoderWeight) + decoderBias)
 decoder_cross_entropy = -tf.reduce_sum(decoderTarget * tf.log(tf.clip_by_value(decoderPrediction,1e-10,1.0)))
 decoder_optimizer = tf.train.AdamOptimizer()
 decoder_minimize = decoder_optimizer.minimize(decoder_cross_entropy)


 #data preperation
 train_input=[]
 train_output=[]
 
 for i in range(len(myList)):
  xi=[]
  xi.append(list(itertools.chain.from_iterable(myList[i])))
  x = np.array(xi)
  
  y= x.astype(np.float)
  train_input.append(y) 
     
  with open("C:/Users/Procheta/Desktop/data/firstWord.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        x = []
        x = [0] * VOCABULARY_SIZE
        x[int(line1[0])] = 1
        train_output.append(x)
        
 #executing the model   
 init_op = tf.global_variables_initializer()
 sess = tf.Session()
 sess.run(init_op)

 batch_size = 1
 no_of_batches =int(int(len(train_input)) / batch_size)
 epoch = NUM_EPOCH_SESSION
 for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
      inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
      ptr+=batch_size
      sess.run(decoder_minimize,{decoderData: inp, decoderTarget: out})
    print ("Epoch For Predicting First Word LSTM: ",str(i))
 #print(sess.run(decoderPrediction,{decoderData: [train_input[0]]}))
 
 l =  sess.run(decoderPrediction,{decoderData: [train_input[0]]})
 flattened = [val for sublist in l for val in sublist]
 xi=[]
 index = flattened.index(max(flattened))
 
 count = 0;
 
 with open("C:/Users/Procheta/Desktop/data/wordId.txt") as fp:
    for line in fp:
        if count == index:
         line1 = line.split(':',1)
         print(line1[0])
        count = count + 1
 
 op=[]
 
 for i in range(len(train_input)):
  l =  sess.run(decoderPrediction,{decoderData: [train_input[i]]})
  flattened = [val for sublist in l for val in sublist]
  index = flattened.index(max(flattened))
  xx = [0]*VOCABULARY_SIZE
  xx[index] = 1
  xxx = np.array(xx)
  yyy = xxx.astype(np.float)
  z=[]
  z.append(yyy)
  op.append(z)
  
 opTest=[]
 wordList = []
 testSecondInput = []
 for i in range(len(testInput)):
  l =  sess.run(decoderPrediction,{decoderData: [testInput[i]]})
  testSecondInput.append(sess.run(state,{decoderData: [testInput[i]]}))
  flattened = [val for sublist in l for val in sublist]
  index = flattened.index(max(flattened))
  x = []
  s = ''
  count = 0
  with open("C:/Users/Procheta/Desktop/data/wordId.txt") as fp:
          for line in fp:
              if count == index:
                  line1 = line.split(':',1)
                  s += line1[0]
                  s += ' '
              count = count + 1 
  x.append(s)         
  wordList.append(x)
  
  
 print("second word")
  
 for i in range(len(testSecondInput)):
  l =  sess.run(decoderPrediction,{decoderData: [testSecondInput[i]]})
  flattened = [val for sublist in l for val in sublist]
  index = flattened.index(max(flattened))
  s = ''
  count = 0
  with open("C:/Users/Procheta/Desktop/data/wordId.txt") as fp:
          for line in fp:
              if count == index:
                  line1 = line.split(':',1)
                  s += line1[0]
                  s += ' '
              count = count + 1        
  wordList[i].append(s)
  
 
 sess.close()

 print(wordList)
 return op, opTest;

def findCosineSimilarity(list1=[], list2=[]):
    x = np.array(list1)
    x1 = x.astype(np.float)
    x11 = np.sort(x1)
    y = np.array(list2)
    y1 = y.astype(np.float)
    y11 = np.sort(y1)
    sim = 0
        
    xindex = 0
    yindex = 0
    while xindex < len(x11) and yindex < len(y11):
        
        if x11[xindex] == y11[yindex] :
            sim = sim + 1
            xindex = xindex + 1
            yindex = yindex + 1
        elif x11[xindex] > y11[yindex] :
            yindex = yindex + 1
            
        elif x11[xindex] < y11[yindex] :
            xindex = xindex + 1
            
    return sim/3
         
 


#running the LSTM for query encoding
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

batch_size = 1
no_of_batches =int(int(len(train_input)) / batch_size)
epoch = NUM_EPOCH_WORD
s=[]
stateArray = []
for i in range(epoch):
    ptr = 0
    stateArray = []
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print ("Epoch For Query Encoding LSTM ",str(i))
    

targetQuery=[]
count = 0
for i in range(NUM_EXAMPLES):
 if count >= NUM_QUERY_SESSION:
     targetQuery.append(train_input[i])
 count = count + 1

queryModelInput=[]
for i in range(NUM_EXAMPLES):
 stateOutput = sess.run(state,{data: [train_input[i]]})
 queryModelInput.append(stateOutput)
 
queryModelOutput=[]
for i in range(len(targetQuery)):
 stateOutput = sess.run(state,{data: [targetQuery[i]]})
 queryModelOutput.append(stateOutput)
 
testDataInput=[]
for i in range(NUM_TEST_EXAMPLES):
 testStateOutput =  sess.run(state,{data: [test_input[i]]})
 testDataInput.append(testStateOutput)


sess.close()
outputData=[]
with tf.variable_scope("scope",reuse = None):
 tf.set_random_seed(42)
 outputData, output_test = newFunc(queryModelInput, testDataInput,queryModelOutput)
 with tf.variable_scope("scope",reuse = None):
  tf.set_random_seed(42)
  output, outputTest  = decoderStage(outputData, output_test)
   