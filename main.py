from deep_Q_learning import * 
from bot_play import *

def main():
    max_episodes = 5000
    
    replay_buffer = deque()
    
    with tf.compat.v1.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size)
        tf.global_variables_initializer().run()
        
        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            
            state = env.reset()
            
            while not done :
                if np.random.rand(1) < e:
                    action = random.randint(0, 242)
                else:
                    # choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))
                    
                # get new state and reward from environment
                next_state, reward, done = env.step(action)
                if done: # big penalty
                    reward = -100
                
                # save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                # buffer 크기 일정하게 유지
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    break
            
            print("Episode: {} steps: {} reward: {}".format(episode, step_count, reward))
            if step_count> 10000:
                pass
            
            if episode % 10 == 1:
                for _ in range(50):
                    # buffer에서 10개 씩 가져옴
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, minibatch)
                print("Loss: ", loss)
        bot_play(mainDQN)
        
        
if __name__ == "__main__":
    main()