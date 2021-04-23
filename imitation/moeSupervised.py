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
imitationDir ="/home/ubuntu/Documents/proj/research/Research/imitation/final_results" 

#pongAct = torch.squeeze(torch.load("imitation/pong_data/act_collection_pong_2.pt")).to(device)
pongObs = torch.squeeze(torch.load("imitation/pong_data/obs_collection_pong_2.pt").to(torch.float))

pongAct1 = torch.squeeze(torch.load("imitation/pong_data/act_split2103_bce_pong_2.pt")).to(device)
###pongAct2 = pongAct1[:, 1]
###pongAct1 = pongAct1[:, 0]


#hcAct = torch.from_numpy(np.load(hcDir+"/acthc.npy")).to(torch.float)
#hcObs = torch.from_numpy(np.load(hcDir+"/obshc.npy")).to(torch.float)
#shuffle = torch.randperm(hcAct.shape[0])
#hcAct = hcAct[shuffle]
#hcObs = hcObs[shuffle]

def train(run_num = 0):
    global pongAct1, pongObs###, pongAct2
    shuffle = torch.randperm(pongAct1.shape[0])
    pongAct1 = pongAct1[shuffle]
    ###pongAct2 = pongAct2[shuffle]
    pongObs = pongObs[shuffle]

    ##model = moeCore.MixtureOfExperts(0, [4]).to(device)
    ##loss_fn = torch.nn.CrossEntropyLoss()
    ##optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    ## pong action space: 6 actions (0,1 = do nothing; 2,4 = up; 3,5 = down)
    ## we only use 4
    model1 = moeCore.MixtureOfExperts(0, [2, 2]).to(device)
    ###loss_fn1 = torch.nn.CrossEntropyLoss()
    loss_fn1 = torch.nn.BCELoss()
    sig = torch.nn.Sigmoid()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)

    ###model2 = moeCore.MixtureOfExperts(0, [2]).to(device)
    ###loss_fn2 = torch.nn.CrossEntropyLoss()
    ###optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)



    #antEnv = gym.make("Ant-v2")
    #halfCheetahEnv = gym.make("HalfCheetah-v2")

    pong = gym.make("PongNoFrameskip-v4")
    pong = make_env(pong)

    loss_hist = []
    #ant_return_hist = []
    pong_return_hist = []
    batch_size = 25
    for j in range(1000):
        ave_loss1 = 0
        ###ave_loss2 = 0
        for i in range(int(pongAct1.shape[0]/batch_size)):
            optimizer1.zero_grad()
            ###optimizer2.zero_grad()

            x = pongObs[i*batch_size:(i+1)*batch_size].to(torch.float).to(device)#torch.cat((antObs[i*batch_size:(i+1)*batch_size], hcObs[i*batch_size:(i+1)*batch_size]), dim=1)
            output1,_,_ = model1(x)
            ###output2,_,_ = model2(x)
            output1 = sig(torch.cat(output1, dim=1)) ### sigmoid because we are using BCELoss
            ###output2 = torch.cat(output2, dim=1)

            target1 = pongAct1[i*batch_size:(i+1)*batch_size]#torch.cat((antAct[i*batch_size:(i+1)*batch_size], hcAct[i*batch_size:(i+1)*batch_size]), dim=1)
            loss1 = loss_fn1(output1, target1)
            loss1 = model1.get_loss(loss1)
            ave_loss1 += loss1.detach()
            loss1.backward()
            optimizer1.step()


            ###target2 = pongAct2[i*batch_size:(i+1)*batch_size]#torch.cat((antAct[i*batch_size:(i+1)*batch_size], hcAct[i*batch_size:(i+1)*batch_size]), dim=1)
            ###loss2 = loss_fn2(output2, target2)
            ###loss2 = model2.get_loss(loss2)
            ###ave_loss2 += loss2.detach()
            ###loss2.backward()
            ###optimizer2.step()

        print("Epoch:", j)
        print("Average Loss:", ave_loss1/int(pongAct1.shape[0]/batch_size))###, ave_loss2/int(pongAct2.shape[0]/batch_size))
        print()
        ## loss_hist.append(ave_loss/int(pongAct.shape[0]/batch_size))


        #antObsTest = antEnv.reset()
        pongObsTest = pong.reset()

        antReturn = 0
        pongReturn = 0
        count_games = 0
        exp_history1 = []
        ###exp_history2 = []

        while count_games < 3:
            obs = get_state(pongObsTest).to(device)#torch.cat((torch.from_numpy(antObsTest), torch.from_numpy(hcObsTest))).to(torch.float)
            output1, _, expert_output1 = model1(obs)
            ###output2, _, expert_output2 = model2(obs)
            exp_history1.append(expert_output1)
            ###exp_history2.append(expert_output2)

            #antObsTest, antReward, antDone, _ = antEnv.step(output[0].detach().cpu().numpy())
            a = np.argmax(output1[0][0].detach().cpu().numpy()) * 2 + np.argmax(output1[1][0].detach().cpu().numpy())
            
            # when 0123 --maped to--> 2103

            if a == 2:
                a = 0
            if a == 0:
                a = 2
            pongObsTest, pongReward, pongDone, _ = pong.step(a)

            #antReturn += antReward
            pongReturn += pongReward
            debug_flag = False
            if pongDone:
                if j == 999:
                    np.save(imitationDir+"/pong_expert_ep999_model1_" + str(count_games) + "_" + str(run_num), np.stack(exp_history1, axis = 0))
                    ###np.save(imitationDir+"/pong_expert_ep999_model2_" + str(count_games) + "_" + str(run_num), np.stack(exp_history2, axis = 0))
                count_games += 1
                pongObsTest = pong.reset()


        #print("Ant Return:",antReturn)
        print("Pong Return:", pongReturn/3.0)
        print()

        #ant_return_hist.append(antReturn)
        pong_return_hist.append(pongReturn/3.0)
    return np.array(loss_hist), np.array(pong_return_hist), model1.gates.prob.detach().cpu().numpy()###, model2.gates.prob.detach().cpu().numpy()


result_id = "_1env_2.0"
loss_result = 0
return_result = 0
repeat=3.0

for i in range(int(repeat)):
    l, r, p1 = train(i)
    ##loss_result += l
    return_result += r
    np.save(imitationDir+"/pong_prob1_"+str(i)+result_id, p1)
    ###np.save(imitationDir+"/pong_prob2_"+str(i)+result_id, p2)
##loss_result /= repeat
return_result /= repeat

##np.save(imitationDir+"/pong_loss"+result_id, loss_result)
#np.save(imitationDir+"/ant_return"+str(result_id), np.array(ant_return_hist))
np.save(imitationDir+"/pong_return"+result_id, return_result)






