import os
submit_dir = os.environ.get('SLURM_SUBMIT_DIR')
with open(f'{submit_dir}/output.txt', 'w') as f:
    f.write('test\n')
