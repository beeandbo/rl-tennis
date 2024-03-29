"""
Program for training and evaling an Multi Agent DDPG within a Unity Tennis environment.

You can tweak hyperparameters, control the number of episodes, etc.

Example: pythonw main.py --episodes 5000 --saveto checkpoint.pth --saveplot scores.png
"""

from unityagents import UnityEnvironment
import maddpg_agent
import coach
import matplotlib.pyplot as plt
import argparse
import numpy as np

def moving_average(input, average_over):
    """Return a weighted moving average over the input.

    Params
    ======
        input (array_like): Weighted average is calculated over these elements
        average_over (integer): Window size of the weighted average
    """
    moving_average_ = 0;
    weight = 1./average_over
    output = []
    for elem in input:
        moving_average_ = moving_average_ * (1-weight) + elem * weight
        output.append(moving_average_)
    return output

def main():
    parser = argparse.ArgumentParser(description='Train a ddpg agent to play the Unity Environment Tennis app')
    parser.add_argument("--episodes", type=int, help="Number of training episodes to run", default=5000)
    parser.add_argument("--max_steps", type=int, help="Maximum steps per episode", default=1000)
    parser.add_argument("--saveto", help="Save agent after training.  agent- and critic- are prepended to the specified name.", default='checkpoint.pth')
    parser.add_argument("--loadfrom", help="Load previously saved model before training")
    parser.add_argument("--min_score", type=float, help="Only save the model if the it achieves this score", default=0.5)
    parser.add_argument("--saveplot", help="Location to save plot of scores")
    parser.add_argument("--environment", help="Path to Unity environment for game (i.e. ./Tennis.App)", default="./Tennis.app")
    parser.add_argument("--eval", type=bool, help="Turns on eval mode, which affects the unity environment and removes the random noise from the predicted agent actions", default=False)
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size
    print("Action size: " + str(action_size))

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)

    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # Create agent and start training
    _agent = maddpg_agent.MADDPGAgent(state_size, action_size, num_agents)
    if args.loadfrom:
        _agent.load(args.loadfrom)
    _coach = coach.Coach(_agent, env)

    # Callback function which will save the agent if it exceeds the
    # minimum score
    max_score = 0
    def save_fn(agent, episode, avg_score):
        nonlocal max_score
        if avg_score > max_score and avg_score > args.min_score and args.saveto:
            agent.save(args.saveto)
            print("Training succeeded.  New max score %s at step %s" % (avg_score, episode))
            max_score = avg_score

    # Train (or eval) the agent
    scores = _coach.run_episodes(args.episodes, args.max_steps, train=not args.eval, callback = save_fn )

    # Plot scores
    plt.plot(scores)
    plt.plot(moving_average(scores, 100), color='red')
    plt.ylabel('Episode scores')
    if args.saveplot:
        plt.savefig(args.saveplot, bbox_inches='tight')

main()
