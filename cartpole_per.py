import gym 
import keras.backend as K 
from keras.models import Model 
from keras.layers import Input, Dense
from keras.optimizers import Adam 
import numpy as np 
import random
import h5py 
from PER import PrioritizedReplay


BATCH_SIZE = 32
EPISODES = 300
MAX_EPS = 1.0
MIN_EPS = 0.001
EPS_DECAY = 0.99
GAMMA = 0.975
LEARNING_RATE = 0.001


class PERAgent():
    def __init__(self, obvs_space, action_space):
        self.obvs = obvs_space
        self.actions = action_space
        self.exploration = MAX_EPS
        self.memory = PrioritizedReplay(2000)
        self.model, self.predict = self._build_predict_model()
        self.target_model = self._build_target_model()
        self.update_target_weights()

    def _build_predict_model(self):
        input_1 = Input(shape=(self.obvs,))
        is_weights = Input([1])
        dense_1 = Dense(64, activation='relu', kernel_initializer='he_normal')(input_1)
        dense_2 = Dense(64, activation='relu', kernel_initializer='he_normal')(dense_1)
        outs = Dense(self.actions, activation='linear')(dense_2)

        def per_loss(y_true, y_pred):
            square_error = K.square(y_true - y_pred)
            return K.mean(is_weights*square_error)

        model = Model(inputs=[input_1, is_weights], outputs=[outs])
        model.compile(loss=per_loss, optimizer=Adam(lr=LEARNING_RATE))

        predict = Model(inputs=[input_1], outputs=[outs])

        return model, predict
    
    def _build_target_model(self):
        input_1 = Input(shape=(self.obvs,))
        dense_1 = Dense(64, activation='relu')(input_1)
        dense_2 = Dense(64, activation='relu')(dense_1)
        outs = Dense(self.actions, activation='linear')(dense_2)

        target = Model(inputs=[input_1], outputs=[outs])
        return target

    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.exploration:
            return random.randrange(self.actions)
        q_values = self.predict.predict(state)
        return np.argmax(q_values[0])

    def store_transition(self, state, action, reward, next_state, done):
        target = self.predict.predict(state)
        target_val = self.target_model.predict(next_state)
        old_target = target[0][action]
        model_action = np.argmax(target)
        if done:
            new_target = reward
        else:
            new_target = reward + GAMMA*target_val[0][model_action]

        error = np.abs(new_target - old_target)
        self.memory.add(error, (state, action, reward, next_state, done))
        self.exploration = max(MIN_EPS, self.exploration*EPS_DECAY)

    def replay_experiences(self):
        if len(self.memory) < BATCH_SIZE:
            return 
        
        batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
        states = []
        targets = []
        errors = []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            model_pred = self.predict.predict(state)
            old_pred = model_pred[0][action]
            next_action = np.argmax(model_pred[0])
            target_pred = self.target_model.predict(next_state)
            q_value = reward 
            if not done:
                q_value += GAMMA*target_pred[0][next_action]
            model_pred[0][action] = q_value
            targets.append(model_pred)
            e = np.abs(old_pred - q_value)
            errors.append(e)

        states = np.array(states)
        states = np.reshape(states, (BATCH_SIZE, self.obvs))
        targets = np.array(targets)
        targets = np.reshape(targets, (BATCH_SIZE, self.actions))
        loss = self.model.train_on_batch([states, is_weights], targets)

        for i in range(BATCH_SIZE):
            ix = idxs[i]
            e = errors[i]
            self.memory.update(ix, e)
        return loss


if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    obvs = env.observation_space.shape[0]
    actions = env.action_space.n 
    agent = PERAgent(obvs, actions)
    
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
            if done and t != 500:
                reward = -10
            next_state = np.reshape(next_state, [1, obvs])
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            state = np.reshape(state, [1, obvs])
            if done:
                print(f'Episode {episode}: Done in t = {t}, exp = {agent.exploration}')
                time_values.append(t)
                #after every episode we update the target models weights
                agent.update_target_weights()               
                break

        
            agent.replay_experiences()

        if np.mean(time_values[-min(10, len(time_values)):]) > 490:
                    agent.model.save_weights("cp_per.h5")
                    print('Solved!')
                    break

        if episode%50 == 0:
            agent.model.save_weights('cp_per.h5')
    val = np.array(time_values)
    print(f'Mean:{np.mean(val)}, Std:{np.std(val)}, Max:{np.max(val)}, Min:{np.min(val)}')


