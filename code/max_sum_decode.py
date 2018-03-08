import numpy 
import decode
from readInput import read_word_indexes,read_test_struct
import matplotlib.pyplot as mp

#X, W, T = data_read.read_decode_input()
#
#y_star = decode.max_sum(X, W, T)
#
#from string import ascii_lowercase
#mapping = dict(enumerate(ascii_lowercase))
#
#for i in range(y_star.shape[0]):
#	print(mapping[y_star[i]])

def word_accuracy(words1,words2):
    count=0
    for i,j in zip(words1,words2):
        if(numpy.array_equal(i,j)):
            count+=1
    return count/len(words1)
    
def get_test_accuracy(y1,y2):
    count=0
    for i,j in zip(y1,y2):
        if(numpy.array_equal(i,j)):
            count+=1
    return count/len(y1)

def form_words(y1,y2):
    word_ends=[]
    word_idx=read_word_indexes()
    word_ends=numpy.insert(numpy.where(word_idx==-1),0,0)
    given_words=[]
    pred_words=[] 
    start=0
    for i in range(len(word_ends)-1):   
        end=word_ends[i+1]+1
        g_word=y1[start:end]
        p_word=y2[start:end]
        start=end
        given_words.append(g_word)
        pred_words.append(p_word)
    return given_words, pred_words 

# Check the decoder with the parameter calculated in 2(b)
def test_params():
    X_test,y_test=read_test_struct()
    y_pred=[]
    a=numpy.loadtxt("best_params",usecols=(0,))
    W=numpy.array(a[:26*128].reshape(26,128))
    T=numpy.array(a[26*128:26*128+26*26].reshape(26,26))
    y_pred = decode.max_sum(X_test, W, T)
    y_pred=[y+1 for y in y_pred]
    y_test=y_test.reshape(26198,)
#    print(y_pred)
#    print(y_test)
    test_acc=get_test_accuracy(y_test,y_pred)
    test_accuracy=(test_acc*100)
    print("Test letter wise accuracy=",test_accuracy)
    y_test=y_test.reshape(len(y_test,))
    given_words, pred_words=form_words(y_test,y_pred)
    w_acc=word_accuracy(given_words,pred_words)
    word_acr =(w_acc*100)
    print("Test word accuracy=",word_acr)


#test_params()

#Test letter wise accuracy= 80.1320711504695
#Test word accuracy= 34.70040721349622


#426 iterations
#Test letter wise accuracy= 80.25040079395373
#Test word accuracy= 34.75858057009889

def plot():
    C=[1,10,100,1000]
    test_accuracy=[44.74,68.80,79.19,80.13]
    word_acr=[1.80,14.89,32.78,34.70]
    mp.figure(1)
    mp.plot(C,test_accuracy)
    mp.title('Letter wise Accuracy vs C - CRF ')
    mp.ylabel('Accuracy')
    mp.xlabel('C')
    mp.figure(2)
    mp.plot(C,word_acr)
    mp.ylabel('Accuracy')
    mp.xlabel('C')  
    mp.title('Word wise Accuracy vs C - CRF ')

#plot()
    
