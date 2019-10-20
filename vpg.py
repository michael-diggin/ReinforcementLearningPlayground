import gym
import keras.backend as K 
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import numpy as np 
import random 
import h5py


EPISODES = 1000

class PolicyAgent():
    def __init__(self, obvs, acts):
        self.obvs_space = obvs
        self.action_space = acts
        self.policy_lr = 0.001
        self.value_lr = 0.001
        self.GAMMA = 0.99
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

        self.policy_net, self.pred_model = self.build_policy_net()
        self.value_net = self.build_value_net()


    def build_policy_net(self):
        input_1 = Input(shape=(self.obvs_space,))
        advantages = Input(shape=[1])
        dense1 = Dense(64, activation='relu', kernel_initializer='he_normal')(input_1)
        dense_2 = Dense(64, activation='relu', kernel_initializer='he_normal')(dense1)
        probs = Dense(self.action_space, activation='softmax')(dense_2)

        def pg_loss(y_true, y_pred):
            model_pred = K.clip(y_pred, 1e-8, 1-1e-8)
            log_probs = -y_true*K.log(model_pred)
            return K.sum(log_probs*advantages)
        
        pol_model = Model(inputs=[input_1, advantages], outputs=[probs])
        pol_model.compile(loss=pg_loss, optimizer=Adam(lr=self.policy_lr))
        pred_model = Model(inputs=[input_1], outputs=[probs])
        return pol_model, pred_model


    def build_value_net(self):
        input_1 = Input(shape=(self.obvs_space,))
        dense1 = Dense(64, activation='relu', kernel_initializer='he_normal')(input_1)
        dense_2 = Dense(64, activation='relu', kernel_initializer='he_normal')(dense1)
        value = Dense(1, activation='linear')(dense_2)
        model = Model(inputs=[input_1], outputs=[value])
        model.compile(loss='mse', optimizer=Adam(lr=self.value_lr))
        return model


    def act(self, state):
        state = state[np.newaxis, :]
        probs = self.pred_model.predict(state)[0]
        value = self.value_net.predict(state)[0]
        policy_action = np.random.choice(self.action_space, 1, p=probs)[0]
        return policy_action, value

    def update_episode_memory(self, state, action, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)

    def get_advantages(self):
        G = np.zeros_like(np.array(self.rewards))
        R = 0
        for i, r in enumerate(self.rewards[::-1]):
            R = r + self.GAMMA*R
            G[-i] = R

        values = np.array(self.values)
        advantages = np.zeros_like(values)
        for i in range(len(values)):
            advantages[i] = G[i] - values[i]
        mean = np.mean(advantages)
        std = np.std(advantages) if np.std(advantages) > 0 else 1

        advantages = (advantages - mean)/std
        return advantages

    def discount_reward(self):
        G = np.zeros_like(np.array(self.rewards))
        R = 0
        for i, r in enumerate(self.rewards[::-1]):
            R = r + self.GAMMA*R
            G[-i] = R 
        return G
        
    
    def train_networks_on_episode(self):
        states = np.array(self.states)
        acts = np.array(self.actions)
        actions = np.zeros([len(acts), self.action_space])
        actions[np.arange(len(acts)), acts] = 1

        advantages = self.get_advantages()
        dis_rewards = self.discount_reward()
        
        pol_loss = self.policy_net.train_on_batch([states, advantages], actions)        
        val_loss = self.value_net.train_on_batch(states, dis_rewards)

        self.states = []
        self.values = []
        self.rewards = []
        self.actions = []
        return pol_loss, val_loss

    def save_weights(self):
        self.policy_net.save_weights('cp_vpq_1.h5')

    def load_policy(self):
        self.policy_net.load_weights('cp_vpq_1.h5')


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_obvs = env.observation_space.shape[0]
    agent = PolicyAgent(n_obvs, n_actions)
    agent.load_policy()

    score_hist = []
    for e in range(20):
        done = False
        state = env.reset()
        t = 0
        while True:
            t += 1
            env.render()
            policy_action, state_value = agent.act(state)
            next_state, reward, done, info = env.step(policy_action)
            reward = reward if not done else -10
            agent.update_episode_memory(state, policy_action, reward, state_value) 
            state = next_state
            if done:
                print(f"Ep: {e}\t timesteps: {t}") 
                score_hist.append(t)
                #_, _ = agent.train_networks_on_episode()
                break

        #if np.mean(score_hist[-10:]) > 475:
        #    print(f"Solved in {e-10} episodes")
        #    agent.save_weights()
        #    break 



    