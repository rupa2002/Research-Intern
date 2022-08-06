import gym
import numpy as np
import random
import simpy
import heapq
from gym import spaces
    
class TelecomEnv:
    """Custom Environment that follows gym interface"""

    def __init__(self,sim_duration=150,alpha=0.2):
        #super(TelecomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(1000)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Discrete(25)
        # Observaton space (resource nodes) is to chose the subarray of 25 blocks of 100(Area of F*T) 
        self.state = [0]*25
        # Action Space to choose 'n' value in the equation t= n*delta
        self.actions=[]
        for x in range(1,1001):
            self.actions.append(x)
        print(self.actions)
        # Demands distribution 
        self.urllc_type=[1]*10
        self.urllc_demands=list(np.random.randint(low=500,high=2000,size=10))
        self.urllc_duration=list(np.random.randint(low=2,high=4,size=10))
        self.urllc=zip(self.urllc_type,self.urllc_demands,self.urllc_duration)
        self.mmtc_type=[2]*20
        self.mmtc_demands=list(np.random.randint(low=100,high=400,size=20))
        self.mmtc_duration=list(np.random.randint(low=3,high=5,size=20))
        self.mmtc=zip(self.mmtc_type,self.mmtc_demands,self.mmtc_duration)
        self.embb_type=[3]*50
        self.embb_demands=list(np.random.randint(low=50,high=100,size=50))
        self.embb_duration=list(np.random.randint(low=5,high=10,size=50))
        self.embb=zip(self.embb_type,self.embb_demands,self.embb_duration)
        self.demands=[]
        self.demands.extend(self.urllc)
        self.demands.extend(self.mmtc)
        self.demands.extend(self.embb)
        # Randomly Shuffling the demands
        random.shuffle(self.demands)
        # Simulation duration is 100 sec
        self.sim_duration = sim_duration
        # InterArrival discrete time distribution
        #self.arrival_distribution=np.random.uniform(0.0,200.0,size=(1,1000)).tolist()
        int_list = random.sample(range(100, 1000), 100)
        self.arrival_time = [x/10 for x in int_list]
        self.arrival_time.sort();
        self.ind=0
        # Next time stop for simpy environment 
        self.next_time_stop=0
        # Resource Wastage and delay 
        self.wastage=0
        self.delay=0
        # Reward calculation parameter 
        self.alpha=alpha
        # Counting the no.of dropped packets
        self.urllc_block=0
        self.mmtc_block=0
        self.embb_block=0
        # Heap data Structure for priority
        self.hq=[]
        # State timer for slices
        self.state_timer=[0]*25
        # Setting up observation and action space sizes
        self.action_size = 1000
        self.observation_size = 25
        
         # simpy environment
        self.env=simpy.Environment()
        print(f"Demands={self.demands}")
        print(f"arrival_time={self.arrival_time}")
    
                    
    def _get_Reward(self):
        """ The Reward is the function of SE(resource wastage & spectral efficiancy) and QoE(delay & reliability component)
                 R= k1*RW + k2*Delay + 10 * urllc_block + 5*(mmtc_block +embb_block)  , k1=0.2,k2=0.8 
             It always returns negative value
                 """
        k1=self.alpha
        k2=1-k1
        loss=k1*self.wastage+k2*self.delay+ 50*self.urllc_block+25*(self.mmtc_block+self.embb_block) 
        return  -1*loss 
                              
    def _matching(self,action):
        '''
        The matching performed here is a non pre-emptive priority Scheduling using heapq datastructure.
        Whenever a node comes it waits for some time t(delay), t=action*delta(delta=0.01), In that time interval  
        if any node is in the arrival distribution it will be pushed into the priority queue.
            In round robin format we will try to schedule and match based on priority and availability 
             1.We will block incase if  delay is greater than 5ms for urllc and 10ms for mmtc & embb
             2.If thats not the case then we will check for available slice, if it's not available,
               then pop and keep it in waiting list other wise allo the slice
               
        '''
        # An empty temporary priority queue
        temp_hq=[]
        while (len(self.hq)>0) :
            temp_hq.append(heapq.heappop(self.hq))
        # Matching all the elements in the waiting queue
        for i in range(len(temp_hq)):
            temp=temp_hq[i][1]*temp_hq[i][2]
            l=-1
            r=-1
            flag=False
            for j in range(25):
                if self.state[j]==0 and l==-1:
                    l=j
                    r=j
                elif self.state[j]==0 and l!=-1:
                    r+=1
                else :
                    l=-1
                    r=-1
                if (r-l+1)*100 >= temp:
                    flag=True
                    break
            if flag :
                self.wastage+= (((r-l+1)*100-temp_hq[i][1]*temp_hq[i][2])/((r-l+1)*100))
                self.delay+= abs((self.env.now+(action+1)*0.01)-temp_hq[i][3])
                for x in range(l,r+1):
                    if x==self.observation_size:
                        continue
                    self.state[x]=1
                    self.state_timer[x]=(self.env.now+action*0.01)+temp_hq[i][2]
            else:
                heapq.heappush(self.hq,temp_hq[i])   
        yield self.env.timeout(action*0.01)
    
    def _clearing_slices(self):
        # Dropping the packets which crossed delay bounds (5ms-URllc & 10ms-emBB,Mmtc)
        temp_hq=self.hq.copy()
        self.hq.clear()
        for i in range(len(temp_hq)):
            if ((temp_hq[i][0]==1 and (self.arrival_time[self.ind]-temp_hq[i][3])>5) or (self.arrival_time[self.ind]-temp_hq[i][3]>10)):
                if(temp_hq[i][0]==1):
                    self.urllc_block+=1
                else :
                    self.mmtc_block+=1
            else:
                heapq.heappush(self.hq,temp_hq[i])
        # Clearing the slices and assigning it to the most prioritized ones in the waiting queue
        l=-1
        r=-1
        for i in range(25):
            if (self.state_timer[i]<=self.arrival_time[self.ind] and self.state[i]==1) :
                self.state[i]=0
                if l==-1 :
                    l=i
                    r=i
                else :
                    r+=1
            else:
                l=-1
                r=-1
            if (len(self.hq)!=0 and (r-l+1)*100>=(self.hq[0][1]*self.hq[0][2])) :
                self.wastage+= (((r-l+1)*100-self.hq[0][1]*self.hq[0][2])/((r-l+1)*100))
                self.delay+= abs(self.state_timer[r]-self.hq[0][3])
                for j in range(l,r+1) :
                    if j==self.observation_size:
                        continue
                    self.state[j]=1
                    self.state_timer[j]=self.hq[0][2]+self.state_timer[j]
                heapq.heappop(self.hq)
                l=-1
                r=-1
        yield self.env.timeout(0)                
    
    def _setting_slices(self,action):
        yield self.env.timeout(self.arrival_time[self.ind]-self.env.now)
        # The arrival of the nodes are pushed into the priority queue 
        while (self.arrival_time[self.ind]<=(self.env.now +(action+1)*0.01) and self.ind<80) :
            temp=list(self.demands[self.ind])
            temp.append(self.arrival_time[self.ind])
            tuple(temp)
            heapq.heappush(self.hq,temp)
            self.ind+=1
        # The _matching function is called to match the resources and demands 
        print(f"matching function is called at :{self.env.now}")
        self.env.process(self._matching(action))
        
        
    def _get_observations(self):
        " Returns current state observatons "
        observations = self.state
        return observations
    
    
    def step(self, action):
        print(f"action:{action}")
        # initiating the reward terms for each step
        self.wastage=0
        self.delay=0
        self.urllc_block=0
        self.mmtc_block=0
        self.embb_block=0
        # __main program__
        self.next_time_stop=self.arrival_time[self.ind]
        if self.ind!=0 :
            self.env.process(self._clearing_slices())
        self.env.process(self._setting_slices(action))
        
        # Running the simulation until 
        if self.ind<80 :
            self.next_time_stop=self.arrival_time[self.ind]+(action+1)*0.01
            
        self.env.run(until=self.next_time_stop)
        
        # Calculating the observation and reward
        observation=self._get_observations()
        
        reward=self._get_Reward()
        
        # Empty info corresponding to the gym environment
        info=dict()
        info[0]=self.delay
        info[1]=self.wastage
        info[2]=self.ind
        #info[4]=(self.demands[self.ind],self.arrival_time[self.ind])
        # Increasing or incrementing the index by 1
        self.ind+=1
        print(f"Current time:{self.env.now}")
        # Check whether terminal state reached (based on sim time)
        terminal = True if self.env.now >= self.sim_duration or self.ind>=80 else False
        return (observation,reward,terminal,info)
    
    def reset(self):
        self.env=simpy.Environment()
        self.ind=0
        # Resetting all the resources
        for x in range(25):
            self.state[x]=0
        # Resetting the wastage,delay & blocked demands
        self.wastage=0
        self.delay=0
        self.urllc_block=0
        self.mmtc_block=0
        self.embb_block=0
        # Returns current state of observations
        observation=self._get_observations()
        return observation  # reward, done, info can't be included
    
    
    def render(self):
        """ display current state"""
        pass
       