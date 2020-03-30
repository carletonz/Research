# this is based off of:
# https://royf.org/pub/pdf/Yan2019Multi.pdf
# https://github.com/AndyLc/mtl-multi-clustering

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

M = 10 # Number of experts
N = 5 # Number of tasks
CONNECTION_SIZE = 10 # output size of expert and input size of task head

# expert where inputs are color images
# (f)
def Expert_conv(input_shape):
    x = keras.Input(shape=input_shape)
    conv1 = keras.layers.Conv2D(6, 5, activation="relu")(x)
    pool1 = keras.layers.MaxPooling2D((2,2), 2)(conv1)
    conv2 = keras.layers.Conv2D(16, 5, activation="relu")(pool1)
    pool2 = keras.layers.MaxPooling2D((2,2), 2)(conv2)
    
    flatten = keras.layers.Flatten()(pool2)
    
    fc1 = keras.layers.Dense(120)(flatten)
    fc2 = keras.layers.Dense(84)(fc1)
    fc3 = keras.layers.Dense(CONNECTION_SIZE)(fc2)
    
    return keras.Model(x, fc3)

# experts where inputs are vectors
# (f)
def Expert_linear(input_shape):
    x = keras.Input(shape=input_shape)
    
    hl1 = keras.layers.Dense(120)(x)
    hl2 = keras.layers.Dense(84)(fc1)
    hl3 = keras.layers.Dense(CONNECTION_SIZE)(fc2)
    
    return keras.Model(x, fc3)

# Decide to use a specific expert on a task
# Weights decide how much each expert matters
# (b / w)
class Gating(keras.layers.Layer):
    def __init__(self):
        super(Gating, self).__init__()
        self.weights = self.add_weight(shape=(M, N), initializer='zeros', trainable=True)
        self.logits = self.add_wieght(shape=(M, N), initializer='zeros', trainable=True)

    def call(self, x, extra_loss):
        bernoulli = tfp.distributions.Bernoulli(logits = self.logits)

        b = bernoulli.sample()
        w = self.weights * b

        logits_loss = tf.math.reduce_sum(bernoulli.log_prob(b), 0)

        output = tf.einsum('mn,bnf->bnf')
        return output, extra_loss + logits_loss

# Task Head for each task
# 'a' is the size of the output space for each task
# (g)
def Task(input_shape):
    x = keras.Input(shape=input_shape)
    
    hl1 = keras.layers.Dense(120)(x)
    hl2 = keras.layers.Dense(84)(fc1)
    hl3 = keras.layers.Dense(CONNECTION_SIZE)(fc2)
    
    return keras.Model(x, fc3)

# (m)
class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, task_output_size, input_type="V"):
        super(MixtureOfExperts, self).__init__()
        self.experts = None
        self.expert_type = None
        if input_type == "I": # image
            self.experts = nn.ModuleList([
                Expert_conv() for i in range(M)
            ])
            self.expert_type = 0
        elif input_type == "V": # vector
            self.experts = nn.ModuleList([
                Expert_linear(input_size) for i in range(M)
            ])
            self.expert_type = 1
        else:
            raise ValueError()
        
        self.gates = Gating()
        self.taskHeads = nn.ModuleList([
                Task(task_output_size[i]) for i in range(N)
            ])
        
        self.train = True
    
    def forward(self, x):
        """
        :param x: tensor containing a batch of images / states / observations / etc.
        :param task: tensor containing a task number
        """
        # makes sure experts are batched and have correct number of input dimentions
        if self.expert_type == 0 and len(x.shape) != 3: # Images
            raise ValueError("Expecting batched images, got {}".format(x.shape))
        if self.expert_type == 1 and len(x.shape) != 2: # Vectors
            raise ValueError("Expecting batched vectors, got {}".format(x.shape))
        
        B = x.shape[0] # batch size
        
        # in: B x width x height
        # out: B x M x F
        expert_output = torch.stack([self.experts[i](x) for i in range(M)], dim = 1)
        
        # in: B x M x F
        # out: B x N x F
        self.cumulative_logits_loss = torch.zeros(N)
        gates_output, self.cumulative_logits_loss = self.gates(expert_output, self.cumulative_logits_loss)
        
        # in: B x N x F
        # out: B x O
        # O is the sum of all task output sizes
        # cancatinate this so all task outputs are in the same dimention
        taskHead_output = torch.cat([self.taskHeads[i](gates_output[:,i]) for i in range(N)], dim=1)
        return taskHead_output
    
    def get_loss(self, loss):
        logits_loss = self.cumulative_logits_loss * loss.detach()
        total_loss = logits_loss + loss
        return total_loss.sum()


if __name__ == "__main__":
    input_type = "V"
    task_output_size = list(range(N))
    input_size = 20
    batches = 50
    output_size = int(sum(task_output_size))
    
    moe = MixtureOfExperts(input_size, task_output_size, input_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(moe.parameters(), lr=0.001, momentum=0.9)
    
    for i in range(5):
        inputs = torch.randn(batches, input_size)
        label = torch.randint(0, output_size, (batches,))

        output = moe(inputs)
        #output, action = torch.max(output, 0)
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss = moe.get_loss(loss)
        loss.backward()
        optimizer.step()
