# # import os
import jax
from absl import flags
# # with open("./checkpoints/pause.txt", "r") as f:
# #     t = f.readline().strip()
# #     print(t[-1])
# #     print(t[:-2])

# # with open('./checkpoints/brc-DMC_DOGS-0/actor.txt', 'rb') as f:
# #     data = f.read()
    
# #     print(len(data))
flags.DEFINE_string('test_flag', 'test_value', 'A test flag.')
FLAGS = flags.FLAGS
print(FLAGS.test_flag)
print(jax.devices())
