import numpy as np
import pytorch.moe_core_torch as moeCore
import torch
import gym
from imitation.wrappers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


hcDir = "/home/ubuntu/Documents/proj/research/Research/imitation/imit-1603432232"
antDir = "/home/ubuntu/Documents/proj/research/Research/imitation/imit-1603343612"
imitationDir ="/home/ubuntu/Documents/proj/research/Research/imitation" 

pongAct = torch.squeeze(torch.load("imitation/act_collection_pong_2.pt")).to(device)
pongObs = torch.squeeze(torch.load("imitation/obs_collection_pong_2.pt").to(torch.float))

#hcAct = torch.from_numpy(np.load(hcDir+"/acthc.npy")).to(torch.float)
#hcObs = torch.from_numpy(np.load(hcDir+"/obshc.npy")).to(torch.float)
#shuffle = torch.randperm(hcAct.shape[0])
#hcAct = hcAct[shuffle]
#hcObs = hcObs[shuffle]

def train():
    global pongAct, pongObs
    shuffle = torch.randperm(pongAct.shape[0])
    pongAct = pongAct[shuffle]
    pongObs = pongObs[shuffle]

    model = moeCore.MixtureOfExperts(0, [4]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    #antEnv = gym.make("Ant-v2")
    #halfCheetahEnv = gym.make("HalfCheetah-v2")

    pong = gym.make("PongNoFrameskip-v4")
    pong = make_env(pong)

    loss_hist = []
    #ant_return_hist = []
    pong_return_hist = []
    batch_size = 50
    for j in range(1000):
        ave_loss = 0
        for i in range(int(pongAct.shape[0]/batch_size)):
            optimizer.zero_grad()
            x = pongObs[i*batch_size:(i+1)*batch_size].to(torch.float).to(device)#torch.cat((antObs[i*batch_size:(i+1)*batch_size], hcObs[i*batch_size:(i+1)*batch_size]), dim=1)
            output, batched = model(x)
            output = torch.cat(output, dim=1)

            target = pongAct[i*batch_size:(i+1)*batch_size]#torch.cat((antAct[i*batch_size:(i+1)*batch_size], hcAct[i*batch_size:(i+1)*batch_size]), dim=1)
            loss = loss_fn(output, target)
            #loss = model.get_loss(loss)
            ave_loss += loss.detach()
            loss.backward()
            optimizer.step()
        print("Epoch:", j)
        print("Average Loss:", ave_loss/int(pongAct.shape[0]/batch_size))
        print()
        loss_hist.append(ave_loss/int(pongAct.shape[0]/batch_size))


        #antObsTest = antEnv.reset()
        pongObsTest = pong.reset()

        antReturn = 0
        pongReturn = 0
        count_games = 0

        while count_games < 5:
            obs = get_state(pongObsTest).to(device)#torch.cat((torch.from_numpy(antObsTest), torch.from_numpy(hcObsTest))).to(torch.float)
            output, batch = model(obs)
            
            #antObsTest, antReward, antDone, _ = antEnv.step(output[0].detach().cpu().numpy())
            pongObsTest, pongReward, pongDone, _ = pong.step(np.argmax(output[0].detach().cpu().numpy()))

            #antReturn += antReward
            pongReturn += pongReward

            if pongDone:
                count_games += 1
                pongObsTest = pong.reset()


        #print("Ant Return:",antReturn)
        print("Pong Return:", pongReturn/5.0)
        print()

        #ant_return_hist.append(antReturn)
        pong_return_hist.append(pongReturn/5.0)
    return np.array(loss_hist), np.array(pong_return_hist)


loss_result = 0
return_result = 0
for i in range(5):
    l, r = train()
    loss_result += l
    return_result += r
loss_result /= 5.0
return_result /= 5.0


result_id = "14-averaged-no-gating-small-task-network-50batch-1000epoch-expert-size-expanded"

np.save(imitationDir+"/pong_loss"+result_id, loss_result)
#np.save(imitationDir+"/ant_return"+str(result_id), np.array(ant_return_hist))
np.save(imitationDir+"/pong_return"+result_id, return_result)






