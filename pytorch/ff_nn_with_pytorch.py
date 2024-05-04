
print("This code will not work, I just put it here for reference. Instead, run pytorch/ff_nn_with_pytorch_chat_gpt_translation.py")
import PyTorch
from PyTorchAug import nn
from PyTorch import np

# randomly initialize our weights with mean 0
net = nn.Sequential()
net.add(nn.Linear(3, 4))
net.add(nn.Sigmoid())
net.add(nn.Linear(4, 1))
net.add(nn.Sigmoid())
net.float()

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]]).astype('float32')
                
y = np.array([[0],
			[1],
			[1],
			[0]]).astype('float32')

crit = nn.MSECriterion()
crit.float()

for j in range(2400):
    
    net.zeroGradParameters()

    # Feed forward through layers 0, 1, and 2
    output = net.forward(X)
    
    # how much did we miss the target value?
    loss = crit.forward(output, y)
    gradOutput = crit.backward(output, y)
    
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    gradInput = net.backward(X, gradOutput)
    
    # lets update our weights
    net.updateParameters(1)
    
    if (j% 200) == 0:
        print("Error:" + str(loss))
