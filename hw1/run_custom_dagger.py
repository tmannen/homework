#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import sys
import tensorflow.contrib.slim as slim

def means_stds(expert_data):
    #mean center and scale
    means = np.mean(expert_data, axis=0)
    stds = np.std(expert_data, axis=0)
    return means, stds

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    print('Creating custom model')
    hidden_size = 128
    expert_data = pickle.load(open("expert_data_" + args.envname, "rb")) #only used for shape in dagger
    x = tf.placeholder(tf.float32, (None,) + expert_data['observations'].shape[1:], name='x') #input
    net = slim.fully_connected(x, hidden_size,  activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.01))
    output = slim.fully_connected(net, expert_data['actions'].shape[-1], activation_fn=None)

    expert_action_placeholder = tf.placeholder(tf.float32, (None, expert_data['actions'].shape[-1]))
    loss = tf.nn.l2_loss(output - expert_action_placeholder)
    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss)

    #sess.run(tf.global_variables_initializer())
    print('Custom model done')

    with tf.Session() as sess:
        tf_util.initialize() #expert model uses this? does this also initialize my model?
        #sess.run(tf.global_variables_initializer()) #custom model, is this needed?

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        expert_actions = []

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            #get expert actions on our observations
            while not done:
                action = sess.run(output, feed_dict={x: obs[None,:]})
                expert_action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                expert_actions.append(expert_action)
                obs, r, done, _ = env.step(action) #take custom model step
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
                    
            returns.append(totalr)

            #then use these actions and observations to train our model
            for epoch in range(epochs):
                for i in range(0, max_observations, batch_size):
                    batch_observations = observations[i:min(i+batch_size, max_observations-1)]
                    batch_expert_actions = expert_actions[i:min(i+batch_size, max_observations-1)]

                    _, loss_score = sess.run([train_op, loss],
                                        feed_dict = {
                                            x : batch_observations,
                                            expert_action_placeholder : batch_expert_actions
                                        }
                                    )

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
