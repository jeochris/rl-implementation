import gym
import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit) # deque 형태로
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n): # tensor 형태로 n개만큼 sample
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s[0])
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128) # state space = 4차원
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2) # action space = 0 (left) or 1 (right)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon): # exploitation만 한다면 그냥 forward만 있어도 됨 but we need exploration
        out = self.forward(obs) # 현재까지 학습된 Q-network 자기 자신에서 forward
        coin = random.random()
        if coin < epsilon: # exploration
            return random.randint(0,1) # random action
        else : # exploitation
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size) # batch size만큼 replay buffer에서 sample

        q_out = q(s) # q_out : batch_size * 2 (left 확률 / right 확률)
        q_a = q_out.gather(1,a) # sample된 a에 해당하는 index에서의 확률 (Q값) 모음
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # q_target으로 s'에서 max 가지는 action에의 Q값 모음
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target) # loss term
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet() # For fixed Q-target
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) # will update Q
    avg_t = 0
    render = False
    for n_epi in range(10000): # for each episode
        if render:
            env = gym.make('CartPole-v1', render_mode = 'human')            
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False
        t = 0
        while not done: # episode 하나 끝날 때까지
            a = q.sample_action(torch.from_numpy(s[0]).float(), epsilon) # 현재 학습중인 Q network에 state input 넣고 e-greedy하게 action 받아옴
            s_prime, r, done, info, dump = env.step(a) # 해당 action으로 step -> next state, reward, done 여부 받아옴
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask)) # 수집된 s,a,r,s' replay buffer에 일단 저장
            s = [s_prime,0]
            score += r # score 기록
            t += 1 # 1초 경과
            if done:
                break
        avg_t += t
        if avg_t/20 > 100:
            print("Solved in {} episodes!".format(n_epi))
            render = True
        if memory.size()>2000: # memory 아직 충분히 차지 않으면 train 안함 (너무 조금 쌓았는데 train 시작하면 처음 sample들이 너무 여러번 추출)
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0: # print interval마다
            q_target.load_state_dict(q.state_dict()) # fixed Q-target 갱신
            print("n_episode :{}, avg_t : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, avg_t/20, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
            avg_t = 0
    env.close()

if __name__ == '__main__':
    main()