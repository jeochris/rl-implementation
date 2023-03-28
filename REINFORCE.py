import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002 # lr 키우면 update 너무 커서 안될수도
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x): # policy NN - parameterized by two linear layers
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item): # r, pi(a|s) appended -> all will be used after trajectory terminates
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]: # starting from recent obtained data
            R = r + gamma * R # G_t
            loss = -torch.log(prob) * R # for grad ascent, apply minus
            loss.backward() # grad descent
            # gd on (- log(pi(a_t|s_t))*G_t)
        self.optimizer.step() # update parameter based on stored on gradient
        self.data = [] # empty current trajectory's data for next trajectory

def main():
    env = gym.make('CartPole-v1')
    pi = Policy() # parameterized policy
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(10000):
        s = env.reset() # observation space = (4,)
        #env.render()
        s = s[0]
        ## print(s)
        done = False
        
        while not done: # fyi, CartPole-v1 forced to terminates at 500 step. -> so max 500
            prob = pi(torch.from_numpy(s).float())
            ## print(prob)
            m = Categorical(prob) # action space = discrete(2)
            a = m.sample() # 현재 2차둰의 prob 기반으로 action sample = 현재 policy 기반으로 action sample 하는 것과 동일
            
            s_prime, r, done, info, dump = env.step(a.item())
            ##print(r, done)
            # r = always 1.0

            pi.put_data((r,prob[a]))
            # policy class의 'data'에 저장 (for whole trajectory)
            # prob(a) = prob 기반으로 sample한게 a -> a sample할 확률 (policy) 얼마였는지
            # why prob[a]도 필요? -> loss term에서 log pi(a|s) * G_t -> pi(a|s)가 prob[a]에 해당

            s = s_prime
            score += r # score는 trajectory에서 총 진행한 timestep수 (이게 return은 아님, no gamma)

            # trajectory 끝날 때까지 진행 (done 뜰때까지)

        ##raise Exception('stop')
        
        ##print(done)
        # 현재 policy로 진행한 trajectory에 대해 policy update!
        # this is not REINFORCE version of updating every timestep
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()