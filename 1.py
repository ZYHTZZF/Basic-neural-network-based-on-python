import pandas as pd
import numpy as np
import scipy.special
import matplotlib.pyplot as plt 



# neural network class definition
class NeuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # self.wih=np.random.rand(self.hnodes,self.inodes)-0.5
        # self.who=np.random.rand(self.onodes,self.hnodes)-0.5
        # 或者可以采用以下更加复杂的方式
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function=lambda x: scipy.special.expit(x)

        # learning rate
        self.lr = learningrate


    # train the neural network
    def train(self, inputs_list, targets_list):
        
        # 把输入转化为二维数组（列向量）
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 计算隐藏层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层的输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最终输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算最终输出层的输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层的误差(target - actual)
        output_errors = targets - final_outputs
        # 计算隐藏层的误差
        hidden_errors = np.dot(self.who.T, output_errors)
        # 更新隐藏层到输出层的权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # 更新输入层到隐藏层的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        # 把输入转化为二维数组（列向量）
        inputs = np.array(inputs_list, ndmin=2).T

        # 计算隐藏层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层的输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最终输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算最终输出层的输出
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

######################################################
# 处理图片
from PIL import Image, ImageOps
import numpy as np

def prepare_image_for_mnist(path):
    from PIL import Image, ImageOps
    import numpy as np

    img = Image.open(path).convert("L")   # 转灰度L；RGB会得到HxWx3，别直接用
    #if img.size != (28, 28):
    #    img = img.resize((28, 28), Image.NEAREST)  # 不是28×28就先缩放（你若已是28×28可去掉）

    arr = np.array(img, dtype=np.uint8)          # 形状：(28, 28)
    arr = 255 - arr                          # 反色，PIL读入的白底黑字，要变成黑底白字
    vec = arr.ravel().astype(np.float32)         # 变成一维向量：(784,)

    # 和你训练一致的缩放到[0.01, 1.0]
    vec = (vec / 255.0) * 0.99 + 0.01

    # 查看处理后的图片
    #img2 = vec.reshape((28, 28))
    #plt.imshow(img2, cmap='grey', interpolation='None')
    #plt.show()


    return vec




######################################################


if __name__ == "__main__":
    # number of input, hidden, output nodes
    input_nodes = 784
    hidden_nodes = 300
    output_nodes = 10

    # learning rate
    learning_rate = 0.3

    print("是否需要重新训练神经网络？")
    ans = input("输入1重新训练，输入0直接测试已有模型：")

    if ans == '1':

    # create instance of neural network
        n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
        data_file=open("./data/mnist_train.csv/mnist_train.csv",'r')
        data_list=data_file.readlines()
        data_file.close()

    #from itertools import islice
    #with open("./mnist_train.csv/mnist_train.csv", "r", encoding="utf-8") as f:
    #    head = list(islice(f, 5))
    #print("前5行：")
    #for i,l in enumerate(head): print(i, l[:120])
    #print("第0行split：", head[0].strip().split(",")[:10], "列数=", len(head[0].strip().split(",")))

    #exit()
    # 查看图片
    #for i in range(10):
    #    all_values=data_list[i].split(',')
    #    image_array=np.asarray(all_values[1:],dtype=np.float32).reshape((28,28))
    #    plt.imshow(image_array,cmap='Greys',interpolation='None')
    #    plt.show()

    # 停止运行

    # 训练神经网络
        for epochs in range(10):
            for e in range(epochs):

                for record in data_list:

                    all_values=record.split(',')

                    scaled_input = (np.asarray(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
                    targets = np.zeros(output_nodes) + 0.01
                    targets[int(all_values[0])] = 0.99
                    n.train(scaled_input, targets)

            test_data_file=open("./mnist_test.csv/mnist_test.csv",'r')
            test_data_list=test_data_file.readlines()
            test_data_file.close()
            scorecard = [] 

            # 查看测试图片
            #for i in range(10):
            #all_values=test_data_list[9].split(',')
            #image_array=np.asarray(all_values[1:],dtype=np.float32).reshape((28,28))
            #plt.imshow(image_array,cmap='Greys',interpolation='None')
            #plt.show()

            #测试神经网络的性能
            for record in test_data_list:
                all_values=record.split(',')
                correct_label=int(all_values[0])
                #print("correct label is ", correct_label)
                scaled_input = (np.asarray(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
                outputs = n.query(scaled_input)
                label = np.argmax(outputs)
                #print("network's answer is ", label)
                #print("\n")
                if (label == correct_label):
                    scorecard.append(1)
                else:
                    scorecard.append(0)
            print("epochs = ", epochs)
            print("performance = ", sum(scorecard)/len(scorecard))
            print("\n")

            # 达到95%以上就停止训练
            if (sum(scorecard)/len(scorecard)) > 0.96:
                break

        # 保存模型
        np.savez_compressed("mnist_nn_model.npz", wih=n.wih, who=n.who)

    else:
        # 直接加载已有模型
        n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
        model = np.load("mnist_nn_model.npz")
        n.wih = model['wih']
        n.who = model['who']

    while True:
        ans = input("输入1测试mnist测试集，输入2测试你自己的图片，输入0退出：")
        if ans == '1':
            test_data_file=open("./data/mnist_test.csv/mnist_test.csv",'r')
            test_data_list=test_data_file.readlines()
            test_data_file.close()
            scorecard = [] 
            for record in test_data_list:
                    all_values=record.split(',')
                    correct_label=int(all_values[0])
                    #print("correct label is ", correct_label)
                    scaled_input = (np.asarray(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
                    outputs = n.query(scaled_input)
                    label = np.argmax(outputs)
                    #print("network's answer is ", label)
                    #print("\n")
                    if (label == correct_label):
                        scorecard.append(1)
                    else:
                        scorecard.append(0)
            
            print("performance = ", sum(scorecard)/len(scorecard))
            print("\n")
        
        elif ans == '2':
        # 你可以用下面的代码测试自己的图片
            while True:
                path = input("input your image path (or 'q' to quit): ")
                if path.lower() == 'q':
                    break
                try:
                    vec = prepare_image_for_mnist(path)  # 读入你自己的图片
                except Exception as e:
                    print(f"Error loading image: {e}")
                    continue
                outputs = n.query(vec)                  # 询问神经网络
                label = np.argmax(outputs)               # 找到最大输出对应的标签
                print("network's answer is ", label)     # 打印结果
                img2 = vec.reshape((28, 28))
                plt.imshow(img2, cmap='Greys', interpolation='None')
                plt.show()
        
        else:
            break
