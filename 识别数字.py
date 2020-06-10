import numpy as np
import scipy.special
import matplotlib.pyplot
from PIL import Image

class Network:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learnrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lrate = learnrate
        self.w1 = np.random.normal(0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.w2 = np.random.normal(0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.act_funcation = lambda x: scipy.special.expit(x)


    def train(self, input_list, target_list) :
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        hide_inputs = np.dot(self.w1, inputs)
        hide_outputs = self.act_funcation(hide_inputs)
        final_inputs = np.dot(self.w2, hide_outputs)
        final_outputs = self.act_funcation(final_inputs)
        output_errors = targets - final_outputs
        hide_errors = np.dot(self.w2.T, output_errors)
        self.w2 += self.lrate*np.dot((output_errors*final_outputs*(1-final_outputs)), np.transpose(hide_outputs))
        self.w1 += self.lrate*np.dot((hide_errors*hide_outputs*(1-hide_outputs)), np.transpose(inputs))


    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hide_inputs = np.dot(self.w1, inputs)
        hide_outputs = self.act_funcation(hide_inputs)
        final_inputs = np.dot(self.w2, hide_outputs)
        final_outputs = self.act_funcation(final_inputs)
        final_outputs = np.around(final_outputs, decimals=3)
        print('图像为：', np.argmax(final_outputs))



inputnodes = 784
hiddennodes = 100
outputnodes = 10
learnrate = 0.3
n = Network(inputnodes, hiddennodes, outputnodes, learnrate)
with open('C:\\Users\\Suspect.X\\Downloads\\mnist_train.csv', 'r') as f:
    test_datas = f.readlines()
for data in test_datas:
    values = data.split(',')
    inputs = np.asfarray(values[1:])/255.0*0.99+0.01
    targets = np.zeros(outputnodes)+0.01
    targets[int(values[0])] = 0.99
    n.train(inputs, targets)
with open('C:\\Users\\Suspect.X\\Downloads\\mnist_test.csv', 'r') as fp:
    test = fp.readlines()
one_test = test[0].split(',')
image = np.asfarray(one_test[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image, cmap='Greys', interpolation='none')
n.query(np.asfarray(one_test[1:])/255*0.99+0.01)
print(one_test[0])


im = Image.open('D:\\a.png')
im = im.convert('L')
array = np.array(im)
array_data = 255.0-array.reshape(784)
array_data = array_data/255.0*0.99+0.01
n.query(array_data)