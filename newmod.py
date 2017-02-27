# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:01:03 2017

@author: Procheta
"""
import numpy as np
import random
from random import shuffle
import tensorflow as tf
import itertools
from scipy import spatial


NUM_EXAMPLES = 100
NUM_TEST_EXAMPLES = 5
VOCABULARY_SIZE = 78
WORDS_PER_QUERY = 3
HIDDEN_LAYER_WORD = 5
HIDDEN_LAYER_QUERY = 5
HIDDEN_LAYER_DECODER = 5
NUM_QUERY_SESSION = 2
NUM_EPOCH_WORD = 5
NUM_EPOCH_QUERY = 5
NUM_EPOCH_SESSION= 5
NUM_EPOCH_NEXTWORD= 5

# load the training query  word Id file 
train_input  = []
with open("C:/users/Procheta/Desktop/inputFile.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        words = line1[0].split(',',WORDS_PER_QUERY)
        tempList=[]
        for word in words:
           tList = ([0]*VOCABULARY_SIZE)
           tList[int(word)] = 1
           x1 = np.array(tList)
           y1 = x1.astype(np.float)
           tempList.append(y1)
        x = np.array(tempList)
        y = x.astype(np.float)
        train_input.append(y)


# load the testing query  word Id file 
test_input=[]
with open("C:/users/Procheta/Desktop/inputFile.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        words = line1[0].split(',', WORDS_PER_QUERY)
        tempList=[]
        for word in words:
           tList = ([0]* VOCABULARY_SIZE)
           tList[int(word)] = 1
           x1 = np.array(tList)
           y1 = x1.astype(np.float)
           tempList.append(y1)
        x = np.array(tempList)
        y = x.astype(np.float)
        test_input.append(y)
        

# load the test output data        
test_output=[]
count = 0
with open("C:/users/Procheta/Desktop/inputFile.txt") as fp:
    for line in fp:
        line1 = line.split('\n')
        if count >= NUM_QUERY_SESSION :
            test_output.append(line1[0])
        count = count + 1
        
#load the target output for the word level LSTM
train_output = []
count = 2
idList=[]
with open("C:/users/Procheta/Desktop/wordIdFile.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        idList.append(line1[0])
        
for i in range(0,NUM_EXAMPLES):
    tList = ([0]*VOCABULARY_SIZE)
    tList[int(idList[i])] = 1;
    train_output.append(tList)

print ("Test and Training data loaded for word level LSTM:")


# Designing the lstm in word level
data = tf.placeholder(tf.float32, [None, WORDS_PER_QUERY,VOCABULARY_SIZE]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, VOCABULARY_SIZE])
num_hidden = HIDDEN_LAYER_WORD
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
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

def newFunc(myList=[], test_input=[]):
    
 #model design   
 queryData = tf.placeholder(tf.float32, [None, NUM_QUERY_SESSION,VOCABULARY_SIZE])
 queryTarget = tf.placeholder(tf.float32, [None, NUM_EXAMPLES])
 queryCell = tf.nn.rnn_cell.LSTMCell(HIDDEN_LAYER_QUERY,state_is_tuple=True)
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
  yi=[0]*NUM_EXAMPLES
  yi[i+2] = 1
  train_output.append(yi)
  
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
     outputData.append(sess.run(queryPrediction,{queryData: [train_input[i]]}))
     
 outputTestData = []
 for i in range(len(test)):
     outputTestData.append(sess.run(queryPrediction,{queryData: [test[i]]}))
 
 
 sess.close()

 return outputData,outputTestData;

#function to predict the first word of the predicted query

def decoderStage(myList=[], testInput=[]):
    
 # designing the model
 decoderData = tf.placeholder(tf.float32, [None, 1, NUM_EXAMPLES])
 decoderTarget = tf.placeholder(tf.float32, [None, VOCABULARY_SIZE])
 decoderCell = tf.nn.rnn_cell.LSTMCell(HIDDEN_LAYER_DECODER,state_is_tuple=True)
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
     
  with open("C:/users/Procheta/Desktop/newIdFile.txt") as fp:
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
 
 with open("C:/users/Procheta/Desktop/wordIdFile1.txt") as fp:
    for line in fp:
        if count == index:
         line1 = line.split(':',1)
         #print(line1[0])
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
 
 for i in range(len(testInput)):
  l =  sess.run(decoderPrediction,{decoderData: [testInput[i]]})
  flattened = [val for sublist in l for val in sublist]
  index = flattened.index(max(flattened))
  xx = [0]* VOCABULARY_SIZE
  xx[index] = 1
  xxx = np.array(xx)
  yyy = xxx.astype(np.float)
  z=[]
  z.append(yyy)
  opTest.append(z)
 
 sess.close()

 return op, opTest;

# function to predict the second word in the target query

def predictNextWord(inputdata=[], testInput=[]):
    
 # designing the model
 decoderData = tf.placeholder(tf.float32, [None, 1, VOCABULARY_SIZE])
 decoderTarget = tf.placeholder(tf.float32, [None, VOCABULARY_SIZE])
 num_hidden_decoder = 5
 decoderCell = tf.nn.rnn_cell.LSTMCell(num_hidden_decoder,state_is_tuple=True)
 val, state = tf.nn.dynamic_rnn(decoderCell, decoderData, dtype=tf.float32)
 val = tf.transpose(val, [1, 0, 2])
 decoderLast = tf.gather(val, int(val.get_shape()[0]) - 1)
 decoderWeight = tf.Variable(tf.truncated_normal([num_hidden, int(decoderTarget.get_shape()[1])]))
 decoderBias = tf.Variable(tf.constant(0.1, shape=[decoderTarget.get_shape()[1]]))
 decoderPrediction = tf.nn.softmax(tf.matmul(decoderLast, decoderWeight) + decoderBias)
 decoder_cross_entropy = -tf.reduce_sum(decoderTarget * tf.log(tf.clip_by_value(decoderPrediction,1e-10,1.0)))
 decoder_optimizer = tf.train.AdamOptimizer()
 decoder_minimize = decoder_optimizer.minimize(decoder_cross_entropy)
 
 #data preperation
 train_input=[]
 train_output=[]
 for i in range(len(inputdata)):
  xi=[]
  xi.append(list(itertools.chain.from_iterable(inputdata[i]))) 
  x = np.array(xi)
  y= x.astype(np.float)
  train_input.append(y) 

 test = []  
 for i in range(len(testInput)):
  xi=[]
  xi.append(list(itertools.chain.from_iterable(testInput[i]))) 
  x = np.array(xi)
  y= x.astype(np.float)
  test.append(y) 
     
  with open("C:/users/Procheta/Desktop/newIdFile.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        x = []
        x = [0] * 78
        x[int(line1[0])] = 1
        train_output.append(x)
        
 #executing the model
 init_op = tf.global_variables_initializer()
 sess = tf.Session()
 sess.run(init_op)

 batch_size = 1
 no_of_batches =int(int(len(train_input)) / batch_size)
 epoch = NUM_EPOCH_NEXTWORD
 for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
      inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
      ptr+=batch_size
      sess.run(decoder_minimize,{decoderData: inp, decoderTarget: out})
    print ("Epoch For Predicting Next Word LSTM:",str(i))
 #print(sess.run(decoderPrediction,{decoderData: [train_input[0]]}))
 l =  sess.run(decoderPrediction,{decoderData: [train_input[0]]})
 flattened = [val for sublist in l for val in sublist]
 xi=[]
 index = flattened.index(max(flattened))
 count = 0;
 
 with open("C:/users/Procheta/Desktop/wordIdFile1.txt") as fp:
    for line in fp:
        if count == index:
         line1 = line.split(':',1)
         #print(line1[0])
        count = count + 1
 
 finalOutput = []    
 for i in range(len(test)):
   l =  sess.run(decoderPrediction,{decoderData: [test[i]]})
   flattened = [val for sublist in l for val in sublist]
   index = flattened.index(max(flattened))
   j1 = 0;
   li = [val for sublist in test[i] for val in sublist]
   xi = []
   for j in range(len(li)):
       if li[j] == 1 :
           j1 = j
   xi.append(j1)
   xi.append(index)
   finalOutput.append(xi)
   
 wordList = []
 for i in range(len(finalOutput)):
     n = np.array(finalOutput[i])
     n1 = n.astype(np.float)
     n11 = []
     n11.append(n1)
     x = [val for sublist in n11 for val in sublist]
     s=''
     for j in range(len(x)):
      count = 0
      with open("C:/users/Procheta/Desktop/wordIdFile1.txt") as fp:
          for line in fp:
              if count == index:
                  line1 = line.split(':',1)
                  s += line1[0]
                  s += ' '
              count = count + 1
              
     wordList.append(s)
     
   
 print(wordList)
 sess.close()
 return finalOutput , wordList;   

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
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
        
    print ("Epoch For Query Encoding LSTM ",str(i))

queryModelInput=[]
for i in range(NUM_EXAMPLES):
 queryModelInput.append(sess.run(prediction,{data: [train_input[i]]}))
 
testDataInput=[]
for i in range(NUM_TEST_EXAMPLES):
 testDataInput.append(sess.run(prediction,{data: [test_input[i]]}))


sess.close()
outputData=[]
with tf.variable_scope("scope",reuse = None):
 outputData, output_test = newFunc(queryModelInput, testDataInput)
 with tf.variable_scope("scope",reuse = None):
  output, outputTest  = decoderStage(outputData, output_test)
  with tf.variable_scope("scope",reuse = None):
   finalOutput, predictedQueryList = predictNextWord(output,outputTest)
   to = []
   for i in range(len(test_output)):
       x = test_output[i].split(',',3)
       xi = np.array(x)
       y = xi.astype(float)
       to.append(y)
       
   testSessionQuery = []
   with open("C:/users/Procheta/Desktop/test_session_file.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        testSessionQuery.append(line1[0])
        
   targetQuery = []
   with open("C:/users/Procheta/Desktop/test_session_file.txt") as fp:
    for line in fp:
        line1=line.split('\n')
        targetQuery.append(line1[0])
   sessionindex = 0;       
   for i in range(len(finalOutput)):
       sim = findCosineSimilarity(finalOutput[i],to[i])
       print('Previous two queries in the session: ')
       print(testSessionQuery[sessionindex])
       print(testSessionQuery[sessionindex + 1])
       sessionindex = sessionindex + 2 
       print('The target Query is: ', targetQuery[i])
       print('The predicted Query is: ',predictedQueryList[i])
       print('The similarity between target and the predicted query is :',sim)