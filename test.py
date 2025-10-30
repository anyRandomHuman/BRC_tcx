import os

checkpoints_dir = '/pfs/work9/workspace/scratch/ka_et4232-tcx/checkpoints'
# checkpoints_dir = 'checkpoints'

for ckp in os.listdir(checkpoints_dir):
    ckp_path = os.path.join(checkpoints_dir, ckp)
    contents = os.listdir(os.path.join(checkpoints_dir, ckp))
    seed_path = ckp_path
    for i in range(99):
        if not f'{i}' in contents:
            seed_path = os.path.join(seed_path, f'{i}')
            os.mkdir(seed_path)
            break
    for c in contents:
        if 'txt' in c:
            os.rename(os.path.join(ckp_path, c), os.path.join(seed_path, c))