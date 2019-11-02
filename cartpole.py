import gym
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import h5py

EPISODES = 1000
BATCH_SIZE = 32
MAX_EPS = 1.0
MIN_EPS = 0.01
EPS_DECAY = 0.995
GAMMA = 0.95
LEARNING_RATE = 0.001

class ReplayMem():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class RLAgent():
    def __init__(self, obvs, actions):
        self.obvs = obvs
        self.actions = actions
        self.exploration_rate = MAX_EPS
        self.memory = ReplayMem(2000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_shape=(self.obvs,), activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model 

    def remember(self, transition):
        self.memory.push(transition)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        else:
            batch = self.memory.sample(BATCH_SIZE)
            for state, action, reward, state_next, done in batch:
                state_next = np.reshape(state_next, [1, self.obvs])
                q_update = reward
                if not done:
                    q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                q_values = self.model.predict(state)
                q_values[0][action] = q_update
                self.model.fit(state, q_values, epochs=1, verbose=0)
            self.exploration_rate *= EPS_DECAY
            self.exploration_rate = max(MIN_EPS, self.exploration_rate)


def is_solved(vals, avg):
    vals = np.array(vals[-100:])
    if avg <= np.mean(vals):
        return True
    else:
        return False


if __name__ == '__main__':

    env = gym.make('CartPole-v0')

    obvs = env.observation_space.shape[0]
    actions = env.action_space.n 
    agent = RLAgent(obvs, actions)
    time_values = []
    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, obvs])
        t = 0
        while True:
            t = t+1
            #env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            state = np.reshape(state, [1, obvs])
            if done:
                print(f'Episode {episode}: Done in t = {t}, exp = {agent.exploration_rate}')
                time_values.append(t)
                break
        
            agent.replay()
        if episode > 100 and is_solved(time_values, 195):
            print('Solved!')
            break
        if episode%50 == 0:
            agent.model.save_weights('.\weights\cp_v0_weights_2.h5')
    val = np.array(time_values)
    print(f'Mean:{np.mean(val)}, Std:{np.std(val)}, Max:{np.max(val)}, Min:{np.min(val)}')

        
