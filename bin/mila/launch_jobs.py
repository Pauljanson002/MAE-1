import os

jobs = [
    ("cos_suk_job.sh",2),
    ("inf_suk_job.sh",1),
    ("cos_long_job.sh",10),
    ("inf_long_job.sh",10)
]

for job, num_sub in jobs:
    for i in range(num_sub):
        os.system(f"sbatch {job}")
