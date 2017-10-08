import subprocess
import multiprocessing as mp

def process_group():

    def run_ps(ps_num):
        subprocess.run(["python", "process_group_with_syncreplica.py", "--job_name=ps", "--task_index={0}".format(ps_num)])
    def run_worker(worker_num):
        subprocess.run(["python", "process_group_with_syncreplica.py", "--job_name=worker", "--task_index={0}".format(worker_num)])

    ps_processes = []
    worker_processes = []
    for i in range(3):
        ps_processes.append(mp.Process(target=run_ps, args=(i, )))
    for i in range(4):
        worker_processes.append(mp.Process(target=run_worker, args=(i, )))

    for p in ps_processes + worker_processes:
        p.start()

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    process_group()
