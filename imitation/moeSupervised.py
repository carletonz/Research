import numpy as np
import pytorch.moe_core_torch as moeCore
import torch
import gym

hcDir = "/home/ubuntu/Documents/proj/research/Research/imitation/imit-1603432232"
antDir = "/home/ubuntu/Documents/proj/research/Research/imitation/imit-1603343612"
imitationDir ="/home/ubuntu/Documents/proj/research/Research/imitation" 

antAct = torch.from_numpy(np.load(antDir+"/actant.npy")).to(torch.float)
antObs = torch.from_numpy(np.load(antDir+"/obsant.npy")).to(torch.float)
shuffle = torch.randperm(antAct.shape[0])
antAct = antAct[shuffle]
antObs = antObs[shuffle]

hcAct = torch.from_numpy(np.load(hcDir+"/acthc.npy")).to(torch.float)
hcObs = torch.from_numpy(np.load(hcDir+"/obshc.npy")).to(torch.float)
shuffle = torch.randperm(hcAct.shape[0])
hcAct = hcAct[shuffle]
hcObs = hcObs[shuffle]


model = moeCore.MixtureOfExperts(hcObs.shape[1], [hcAct.shape[1]])
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

antEnv = gym.make("Ant-v2")
halfCheetahEnv = gym.make("HalfCheetah-v2")

loss_hist = []
ant_return_hist = []
hc_return_hist = []
batch_size = 500
for j in range(10000):
    ave_loss = 0
    for i in range(int(antAct.shape[0]/batch_size)):
        optimizer.zero_grad()
        x = hcObs[i*batch_size:(i+1)*batch_size].to(torch.float)#torch.cat((antObs[i*batch_size:(i+1)*batch_size], hcObs[i*batch_size:(i+1)*batch_size]), dim=1)
        output, batched = model(x)
        output = torch.cat(output, dim=1)

        target = hcAct[i*batch_size:(i+1)*batch_size]#torch.cat((antAct[i*batch_size:(i+1)*batch_size], hcAct[i*batch_size:(i+1)*batch_size]), dim=1)
        loss = loss_fn(output, target)
        loss = model.get_loss(loss)
        ave_loss += loss.detach()
        loss.backward()
        optimizer.step()
    print("Epoch:", j)
    print("Average Loss:", ave_loss/int(antAct.shape[0]/batch_size))
    print()
    loss_hist.append(ave_loss/int(antAct.shape[0]/batch_size))


    #antObsTest = antEnv.reset()
    hcObsTest = halfCheetahEnv.reset()

    antReturn = 0
    hcReturn = 0

    for i in range(4000):
        obs = torch.from_numpy(hcObsTest).to(torch.float)#torch.cat((torch.from_numpy(antObsTest), torch.from_numpy(hcObsTest))).to(torch.float)
        output, batched = model(obs)

        #antObsTest, antReward, antDone, _ = antEnv.step(output[0].detach().cpu().numpy())
        hcObsTest, hcReward, hcDone, _ = halfCheetahEnv.step(output[0].detach().cpu().numpy())

        #antReturn += antReward
        hcReturn += hcReward

    #print("Ant Return:",antReturn)
    print("Cheetah Return:", hcReturn)
    print()

    #ant_return_hist.append(antReturn)
    hc_return_hist.append(hcReturn)

result_id = 2

np.save(imitationDir+"/loss"+str(result_id), np.array(loss_hist))
#np.save(imitationDir+"/ant_return"+str(result_id), np.array(ant_return_hist))
np.save(imitationDir+"/hc_return"+str(result_id), np.array(hc_return_hist))






