import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle
import tf_util
import sys
import numpy as np
import load_policy

#TODO: take observations from our untrained model. Use expert model to label these observations

def means_stds(expert_data):
    #mean center and scale
    means = np.mean(expert_data, axis=0)
    stds = np.std(expert_data, axis=0)
    return means, stds

def train(sess, output, data, epochs):
    for epoch in range(epochs):
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    hidden_size = 128
    batch_size = args.batchsize

    ###Create model
    x = tf.placeholder(tf.float32, (None,) + expert_data['observations'].shape[1:], name='x') #input
    net = slim.fully_connected(x, hidden_size,  activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.01))
    output = slim.fully_connected(net, expert_data['actions'].shape[-1], activation_fn=None)

    expert_action_placeholder = tf.placeholder(tf.float32, (None, expert_data['actions'].shape[-1]))
    loss = tf.nn.l2_loss(output - expert_action_placeholder)
    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps

    returns = []
    observations = []
    actions = []
    expert_actions = []
    eps = 1e-6 #used for preprocessing, in case std is 0

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0

        while not done:
            action = sess.run(output, feed_dict={x: obs[None,:]}) #run with our model
            expert_action = policy_fn(obs[None, :])
            observations.append(obs)
            actions.append(action)
            expert_actions.append(expert_action)
            obs, r, done, _ = env.step(action) #take the action the untrained model does to step into unexplored area
            totalr += r
            steps += 1
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