# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:00:04 2018

@author: ashwa
"""

import subprocess
test_accuracy =[]
import matplotlib.pyplot as mp
C=[1,10,100,1000]
output = subprocess.getoutput('svm_hmm_learn -c 1 train_struct.txt declaration.model')
output = subprocess.getoutput('svm_hmm_classify test_struct.txt declaration.model test.outtags ')
index=output.find("Zero/one-error on test set")
test_accuracy.append(100-float(output[index+28:index+28+4]))

output1 = subprocess.getoutput('svm_hmm_learn -c 10 train_struct.txt declaration.model')
output1 = subprocess.getoutput('svm_hmm_classify test_struct.txt declaration.model test.outtags ')
index=output1.find("Zero/one-error on test set")
test_accuracy.append(100-float(output1[index+28:index+28+4]))

output2 = subprocess.getoutput('svm_hmm_learn -c 100 train_struct.txt declaration.model')
output2 = subprocess.getoutput('svm_hmm_classify test_struct.txt declaration.model test.outtags ')
index=output2.find("Zero/one-error on test set")
test_accuracy.append(100-float(output2[index+28:index+28+4]))

output3 = subprocess.getoutput('svm_hmm_learn -c 1000 train_struct.txt declaration.model')
output3 = subprocess.getoutput('svm_hmm_classify test_struct.txt declaration.model test.outtags ')
index=output.find("Zero/one-error on test set")
test_accuracy.append(100-float(output3[index+28:index+28+4]))

#output = subprocess.call(["svm_hmm_learn", "-c", str(C),"train_struct.txt declaration.model"])
#process = subprocess.run(["svm_hmm_learn", "-c", str(C),"train_struct.txt declaration.model"],stdout=subprocess.PIPE)
#process = subprocess.Popen(["svm_hmm_learn", "-c", str(C),"train_struct.txt declaration.model"], stdout=subprocess.PIPE)
#stdout = process.communicate()[0]
#print ('STDOUT:{}'.format(stdout))

mp.plot(C,test_accuracy)


