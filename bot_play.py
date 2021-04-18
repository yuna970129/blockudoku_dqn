from deep_Q_learning import *

def bot_play(mainDQN):
    # see our trained network in action
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break