import gym
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import h5py

EPISODES = 300
BATCH_SIZE = 64
MAX_EPS = 1.0
MIN_EPS = 0.001
EPS_DECAY = 0.99
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



class DDQNAgent():
    def __init__(self, obvs_space, action_space):
        self.obvs_space = obvs_space
        self.action_space = action_space
        self.exploration_rate = MAX_EPS
        self.memory = ReplayMem(2000)
        self.buffer_size = 1000
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_weights()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.obvs_space,), activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self.action_space, activation='linear', kernel_initializer='he_normal'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, transition):
        self.memory.push(transition)
        self.exploration_rate = self.exploration_rate*EPS_DECAY
        self.exploration_rate = max(self.exploration_rate, MIN_EPS)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.buffer_size:
            return 
        batch = self.memory.sample(BATCH_SIZE)
        states = []
        targets = []
        for state, action, reward, next_state, done in batch:
            state = np.reshape(state, (1, self.obvs_space))
            states.append(state)
            next_state = np.reshape(next_state, (1, self.obvs_space))

            #this is where the DDQN algorithm comes into play
            q_values = self.model.predict(state)
            model_next_predict = self.model.predict(next_state)
            target_predict = self.target_model.predict(next_state)

            action_selection = np.argmax(model_next_predict[0])
            q_update = reward
            if not done:
                q_update = reward + GAMMA*(target_predict[0][action_selection])

            q_values[0][action] = q_update
            targets.append(q_values)
        #fit the model on the new q value
        states = np.array(states)
        states = np.reshape(states, (BATCH_SIZE, self.obvs_space))
        targets = np.array(targets)
        targets = np.reshape(targets, (BATCH_SIZE, self.action_space))
        self.model.fit(states, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)


if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    obvs = env.observation_space.shape[0]
    actions = env.action_space.n 
    agent = DDQNAgent(obvs, actions)
    
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
            reward = reward if not done else - 100
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            state = np.reshape(state, [1, obvs])
            if done:
                print(f'Episode {episode}: Done in t = {t}, exp = {agent.exploration_rate}')
                time_values.append(t)
                #after every episode we update the target models weights
                agent.update_target_weights()

                
                
                break

        
            agent.replay()

        if np.mean(time_values[-min(10, len(time_values)):]) > 490:
                    agent.model.save_weights(".\weights\cp_ddqn_2.h5")
                    print('Solved!')
                    break

        if episode%50 == 0:
            agent.model.save_weights('.\weights\cp_ddqn_2.h5')
    val = np.array(time_values)
    print(f'Mean:{np.mean(val)}, Std:{np.std(val)}, Max:{np.max(val)}, Min:{np.min(val)}')
    

        

