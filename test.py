# # import os
# import jax
# # with open("./checkpoints/pause.txt", "r") as f:
# #     t = f.readline().strip()
# #     print(t[-1])
# #     print(t[:-2])

# # with open('./checkpoints/brc-DMC_DOGS-0/actor.txt', 'rb') as f:
# #     data = f.read()
    
# #     print(len(data))
# print(jax.devices('gpu'))
# # a = jax.numpy.zeros((3,3))
# # print(a.device)

website = '3e7a8db199c7e791e341e6d388d078bfc6b7a77c6f10282edf9c7b82f84f3ad4'
mine = '3e7a8db199c7e791e341e6d388d078bfc6b7a77c6f10282edf9c7b82f84f3ad4'
print(website == mine)