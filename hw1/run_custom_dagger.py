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
    means = np.mean(expert_data, axis=0)
    stds = np.std(expert_data, axis=0)
    return means, stds

def train(tf_model_param, data, epochs, batch_size):
    """

    :param tf_model_param: The variables for tensorflow training use
    :param data: observations and actions
    :param epochs: how many epochs to train
    :param batch_size: batch size
    :return:
    """
    observations = data['observations']
    actions = data['actions']
    train_op = tf_model_param['train_op']
    sess = tf_model_param['sess']
    x = tf_model_param['x']
    expert_action_placeholder = tf_model_param['expert_action_placeholder']
    loss = tf_model_param['loss']

    max_observations = len(observations)
    losses = []

    for epoch in range(epochs):
        for i in range(0, max_observations, batch_size):
            batch_observations = observations[i:min(i + batch_size, max_observations - 1)]
            batch_expert_actions = actions[i:min(i + batch_size, max_observations - 1)]
            batch_expert_actions = batch_expert_actions.reshape(
                (batch_expert_actions.shape[0], batch_expert_actions.shape[-1]))

            _, loss_score = sess.run([train_op, loss],
                                     feed_dict={
                                         x: batch_observations,
                                         expert_action_placeholder: batch_expert_actions
                                     }
                                     )

            losses.append(loss_score)

        print(np.mean(losses))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument('--preprocess', action='store_true') #cant preprocess because expert model takes non preprocessed data?
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    print('Creating custom model')
    hidden_size = 128
    expert_data = pickle.load(open("expert_data_" + args.envname, "rb")) #train first with this?
    x = tf.placeholder(tf.float32, (None,) + expert_data['observations'].shape[1:], name='x') #input
    net = slim.fully_connected(x, hidden_size,  activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001))
    output = slim.fully_connected(net, expert_data['actions'].shape[-1], activation_fn=None)

    expert_action_placeholder = tf.placeholder(tf.float32, (None, expert_data['actions'].shape[-1]))
    loss = tf.nn.l2_loss(output - expert_action_placeholder)
    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(loss)

    expert_means, expert_stds = means_stds(expert_data['observations'])
    print('Custom model done')

    with tf.Session() as sess:
        tf_util.initialize() #expert model uses this? does this also initialize my model? answer: yes

        tf_model_param = {
            'loss': loss,
            'train_op': train_op,
            'expert_action_placeholder': expert_action_placeholder,
            'x': x,
            'sess' : sess
        }

        #first train some with expert data
        eps = 1e-6
        if args.preprocess:
            expert_data['observations'] = (expert_data['observations'] - expert_means) / (expert_stds + eps)

        all_observations = expert_data['observations']
        all_actions = expert_data['actions']
        train(tf_model_param, expert_data, args.epochs, args.batch_size)

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []

        for i in range(args.num_rollouts):
            print('iter', i)
            observations = []
            actions = []
            expert_actions = []

            obs = env.reset()
            if args.preprocess:
                obs = (obs - expert_means) / (expert_stds + eps)
            print(obs.shape)
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
                if args.preprocess:
                    obs = (obs - expert_means) / (expert_stds + eps)

                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
                    
            returns.append(totalr)
            print(totalr)

            all_observations = np.concatenate([all_observations, np.array(observations)])
            print("Size of observation data: ", all_observations.shape)
            all_actions = np.concatenate([all_actions, np.array(expert_actions)])
            expert_means, expert_stds = means_stds(all_observations)

            new_data = {
                'observations' : all_observations,
                'actions' : all_actions
            }

            train(tf_model_param, new_data, args.epochs, args.batch_size)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
