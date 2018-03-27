# Multi-layer-perceptron
## Description

In this problem, we use a network with 4 binary input elements, 4 hidden units for the first layer, and one output unit for the second layer. The desired output is 1 if there are an odd number of 1s in the input elements, and otherwise 0.  
Generally, we use learning rate from 0.05 to 0.5.

## Usage  
python MLP.py learningRate (momentum)  
###### if the momentum is not specifed, we will use the default value 0.


## Results 
We record the number of iterations to converge for different learning rates and momentum.  
|Learning Rate      |0.05    |0.1    |0.15  |0.2   |0.25  |0.3   |0.35  |0.4   |0.45  |0.5  |
|--------------     |----    |-----  |---   |---   |---   |---   |---   |---   |---   |---  |
|Momentum = 0       |1358782 |867793|340095 |71287 |44794 |42089 |13900 |38008 |60049 |29524|
|Momentum = 0.9     |282183  |63251 |10531  |7382  |5463  |4363  |4119  |3144  |30689 |2500 |

## Analysis

#### 1. If the learning rate is too small or too large, it will converge slowly. 
For the same momentum (eg. momentum = 0), when the learning rate is 0.35, the number of epochs is the minimum among all choices. And when the learning rate is too small, the number of epochs is so big. When the learning rate is 0.4, the number of epochs begin to increase.  

#### 2. As the momentum is applied, it converges more quickly.  
We can find that with the momentum = 0.9, the number of epochs decreases a lot. When the learning rate is 0.45, the number of epochs can be 3144.  
In conclusion, we can find that  
(1) When the learning rate is too large or too small, it converges slowly. Having an appropriate learning rate is very important.  
(2) When momentum is applied, the number of epochs decreases a lot. But the effects of the momentum will not hide the effects of the learning rate --- when the momentum is the same, the one with a too small or too large learning rate will always converge slower.
