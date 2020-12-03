from baselines.worker import worker
from envs.gazebo import GazeboEnv

env = GazeboEnv()
worker = worker(env)

raw_input()

worker.work()
