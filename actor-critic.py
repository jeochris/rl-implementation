import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4,256) # policy, value have common initial fc layer - current state representation layer 공유하는 느낌
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1) # value output = just one value
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0): # parameterized polciy
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x): # parameterized value
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self): # batch? -> similar with A2C! (n_rollout samples into a batch)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = [] # clear batch
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        ##print(s, a, r, s_prime, done)

        td_target = r + gamma * self.v(s_prime) * done # target param is fixed within batch, so its fine (update all params at last time)
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        ##print(pi)
        pi_a = pi.gather(1,a) # obtain pi(a|s) value
        ##print(pi_a)
        ##raise Exception('stop')

        # loss term together -> ascent with actor loss / descent with critic loss
        # detach for no grad update on delta, td_target (each should be considered as const in each term)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         
      
def main():  
    env = gym.make('CartPole-v1')
    model = ActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        s = env.reset() # observation space = (4,)
        s = s[0]

        done = False

        while not done:
            # why n_rollout?
            # actor-critic can update parameter within trajectory (no G_t) -> can update more frequently (by n_rollout)
            # <-> REINFORCE : should update after trajectory (use G_t)
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob) # action space = discrete(2)
                a = m.sample() # 현재 2차둰의 prob 기반으로 action sample = 현재 policy 기반으로 action sample 하는 것과 동일
                s_prime, r, done, info, dump = env.step(a.item()) # r = always 1.0

                model.put_data((s,a,r,s_prime,done)) # collected sample
                
                s = s_prime
                score += r # score는 trajectory에서 총 진행한 timestep수 (이게 return은 아님, no gamma)
                
                if done:
                    break                     
            
            # after rollout, update
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()