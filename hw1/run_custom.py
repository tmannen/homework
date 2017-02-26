import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle
import tf_util
import sys
import numpy as np

#TODO: mean center observations. More expert observations, validation?
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    hidden_size = 128
    batch_size = args.batchsize
    expert_data = pickle.load(open("expert_data_" + args.envname, "rb"))

    ###Create model
    x = tf.placeholder(tf.float32, (None,) + expert_data['observations'].shape[1:], name='x') #input
    net = slim.fully_connected(x, hidden_size, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.01))
    output = slim.fully_connected(net, expert_data['actions'].shape[-1], activation_fn=None)

    expert_action_placeholder = tf.placeholder(tf.float32, (None, expert_data['actions'].shape[-1]))
    loss = tf.nn.l2_loss(output - expert_action_placeholder)
    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    observations = expert_data['observations']
    expert_actions = expert_data['actions']
    expert_actions = expert_actions.reshape((expert_actions.shape[0], expert_actions.shape[-1]))

    max_observations = expert_actions.shape[0]
    for epoch in range(100):
        losses = []
        for i in range(0, max_observations, batch_size):

            batch_observations = observations[i:min(i+batch_size, max_observations-1)]
            batch_expert_actions = expert_actions[i:min(i+batch_size, max_observations-1)]

            _, loss_score = sess.run([train_op, loss],
                                feed_dict = {
                                    x : batch_observations,
                                    expert_action_placeholder : batch_expert_actions
                                }
                            )

            losses.append(loss_score)

        print(np.mean(losses))


    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps
    print(max_steps)
    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0

        while not done:
            action = sess.run(output, feed_dict={x: obs[None,:]}) #run with our model
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            print(done)
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    main()