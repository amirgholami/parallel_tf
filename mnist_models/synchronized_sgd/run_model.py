import subprocess
import multiprocessing as mp

def synchronized_model():

    def run_ps():
        subprocess.run(["python", "synchronized_sgd.py",
                "--worker_hosts=localhost:22223,localhost:22224", "--job_name=ps", "--task_index=0"])
    def run_worker(worker_num):
        subprocess.run(["python", "synchronized_sgd.py",
                "--worker_hosts=localhost:22223,localhost:22224", "--job_name=worker", "--task_index={0}".format(worker_num)])

    ps_process = mp.Process(target=run_ps)
    worker_processes = []
    for i in range(2):
        worker_processes.append(mp.Process(target=run_worker, args=(i, )))

    worker_processes[0].start()
    worker_processes[1].start()
    ps_process.start()

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    synchronized_model()
