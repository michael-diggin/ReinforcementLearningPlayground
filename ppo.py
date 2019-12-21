import gym
import keras.backend as K 
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import numpy as np 
import random 
import h5py
import tensorflow as tf


EPISODES = 3000

class PPOAgent():
    def __init__(self, obvs, acts):
        self.obvs_space = obvs
        self.action_space = acts
        self.policy_lr = 0.001
        self.value_lr = 0.0005
        self.GAMMA = 0.99
        self.LAMBDA = 0.97
        self.clipping = 0.15
        self.ENT_FACTOR = 0.0001
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.old_preds = []

        self.policy_net, self.pred_model = self.build_policy_net()
        self.value_net = self.build_value_net()


    def build_policy_net(self):
        input_1 = Input(shape=(self.obvs_space,))
        advantages = Input(shape=[1])
        old_pred = Input(shape=(self.action_space,))
        dense1 = Dense(64, activation='relu', kernel_initializer='he_normal')(input_1)
        dense2 = Dense(32, activation='relu', kernel_initializer='he_normal')(dense1)
        probs = Dense(self.action_space, activation='softmax')(dense2)

        def ppo_loss(y_true, y_pred):
            new_prob = y_true*y_pred
            old_prob = y_true*old_pred
            r = new_prob/(old_prob + 0.00001)
            clipped = K.clip(r, 1-self.clipping, 1+self.clipping)*advantages
            loss = -K.mean(K.minimum(r*advantages, clipped))
            entropy_loss = -K.mean(self.ENT_FACTOR*new_prob*K.log(new_prob + 1e-10))
            return loss + entropy_loss
        
        pol_model = Model(inputs=[input_1, advantages, old_pred], outputs=[probs])
        pol_model.compile(loss=ppo_loss, optimizer=Adam(lr=self.policy_lr))
        pred_model = Model(inputs=[input_1], outputs=[probs])
        return pol_model, pred_model


    def build_value_net(self):
        input_1 = Input(shape=(self.obvs_space,))
        dense1 = Dense(32, activation='relu', kernel_initializer='he_normal')(input_1)
        value = Dense(1, activation='linear')(dense1)
        model = Model(inputs=[input_1], outputs=[value])
        model.compile(loss='mse', optimizer=Adam(lr=self.value_lr))
        return model


    def act(self, state):
        state = state[np.newaxis, :]
        p = self.pred_model.predict(state)
        probs = p[0]
        value = self.value_net.predict(state)[0]
        policy_action = np.random.choice(self.action_space, 1, p=probs)[0]
        self.old_preds.append(p[0])
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
        
    def get_gae(self):
        advantages = []
        advantage = 0
        next_value = 0
        for r, v in zip(reversed(self.rewards), reversed(self.values)):
            td_error = r + next_value * self.GAMMA - v
            advantage = td_error + advantage * self.GAMMA * self.LAMBDA
            next_value = v
            advantages.insert(0, advantage)
        
        advantages = np.array(advantages)
        std = max(np.std(advantages), 0.001)
        advantages = (advantages - advantages.mean()) / std
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
        old_preds = np.array(self.old_preds)
                
        pol_loss = self.policy_net.train_on_batch([states, advantages, old_preds], actions)      
        val_loss = self.value_net.train_on_batch(states, dis_rewards)
        
        self.states = []
        self.values = []
        self.rewards = []
        self.actions = []
        self.old_preds = []
        return pol_loss, val_loss

    def save_weights(self):
        self.policy_net.save_weights('.\weights\lunarlander_ppo_2.h5')
        self.value_net.save_weights('.\weights\lunarlander_value_weights_2.h5')

    def load_policy(self):
        self.policy_net.load_weights('.\weights\lunarlander_ppo_2.h5')
        self.value_net.load_weights('.\weights\lunarlander_value_weights_2.h5')


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')

    #log_dir = 'lunarlander/logs'
    #summary_writer = tf.summary.create_file_writer(log_dir)

    n_actions = env.action_space.n
    n_obvs = env.observation_space.shape[0]
    agent = PPOAgent(n_obvs, n_actions)
    agent.load_policy()

    score_hist = []
    for e in range(EPISODES):
        done = False
        state = env.reset()
        t = 0
        total_reward = 0
        while True:
            t += 1
            env.render()
            policy_action, state_value = agent.act(state)
            next_state, reward, done, info = env.step(policy_action)
            total_reward += reward
            agent.update_episode_memory(state, policy_action, reward, state_value) 
            state = next_state
            if done:
                score_hist.append(total_reward)
                print(f"Ep: {e}\t Episode Length: {t}\t Reward: {total_reward}\t Reward_100: {np.mean(score_hist[-100:])}") 
                
                #pol_loss, val_loss = agent.train_networks_on_episode()

                #with summary_writer.as_default():
                #    tf.summary.scalar('Reward', total_reward, step=e)
                #    tf.summary.scalar('Reward_100', np.mean(score_hist[-100:]), step=e)
                #    tf.summary.scalar('Policy Loss', pol_loss, step=e)
                #    tf.summary.scalar('Value Loss', val_loss, step=e)

                break

        #if (len(score_hist) >= 100) and (np.mean(score_hist[-100:]) >= 200):
        #    print(f"Solved in {e-10} episodes")
        #    agent.save_weights()
            #summary_writer.close()
        #    break 

        #if e%50 == 0:
        #    agent.save_weights()