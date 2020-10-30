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


model = moeCore.MixtureOfExperts(antObs.shape[1]+hcObs.shape[1], [antAct.shape[1], hcAct.shape[1]])
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

antEnv = gym.make("Ant-v2")
halfCheetahEnv = gym.make("HalfCheetah-v2")

loss_hist = []
ant_return_hist = []
hc_return_hist = []

for j in range(100):
    ave_loss = 0
    for i in range(int(antAct.shape[0]/50)):
        optimizer.zero_grad()
        x = torch.cat((antObs[i*50:(i+1)*50], hcObs[i*50:(i+1)*50]), dim=1)
        output, batched = model(x)
        output = torch.cat(output, dim=1)

        target = torch.cat((antAct[i*50:(i+1)*50], hcAct[i*50:(i+1)*50]), dim=1)
        loss = loss_fn(output, target)
        loss = model.get_loss(loss)
        ave_loss += loss.detach()
        loss.backward()
        optimizer.step()
    print("Epoch:", j)
    print("Average Loss:", ave_loss/int(antAct.shape[0]/50))
    print()
    loss_hist.append(int(ave_loss/int(antAct.shape[0]/50)))


    antObsTest = antEnv.reset()
    hcObsTest = halfCheetahEnv.reset()

    antReturn = 0
    hcReturn = 0

    for i in range(4000):
        obs = torch.cat((torch.from_numpy(antObsTest), torch.from_numpy(hcObsTest))).to(torch.float)
        output, batched = model(obs)

        antObsTest, antReward, antDone, _ = antEnv.step(output[0].detach().cpu().numpy())
        hcObsTest, hcReward, hcDone, _ = halfCheetahEnv.step(output[1].detach().cpu().numpy())

        antReturn += antReward
        hcReturn += hcReward

    print("Ant Return:",antReturn)
    print("Cheetah Return:", hcReturn)
    print()

    ant_return_hist.append(antReturn)
    hc_return_hist.append(hcReturn)

np.save(imitationDir+"/loss", np.array(loss_hist))
np.save(imitationDir+"/ant_return", np.array(ant_return_hist))
np.save(imitationDir+"/hc_return", np.array(hc_return_hist))






