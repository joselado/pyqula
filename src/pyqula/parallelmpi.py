
#def pcallgpu(f,xs):
#    """Pcall using MPI"""
#    from mpi4py import MPI
#    import torch
#    
#    # Initialize MPI
#    comm = MPI.COMM_WORLD
#    rank = comm.Get_rank()
#    size = comm.Get_size()
#    
#    # Assign GPUs based on process rank
#    torch.cuda.set_device(rank % torch.cuda.device_count())
#    
#    # Function to perform CUDA-based computation
#    def cuda_compute(x):
#        device = torch.device(f'cuda:{rank}')
#        return f(x)
#    
#    # Each process will work on a different part of the data
#    if rank == 0:
#        data = xs # Create data on CPU
#    else:
#        data = None
#    
#    # Scatter the data to all processes
#    data = comm.scatter(data, root=0)
#    
#    # Perform CUDA computation on each process's data
#    local_result = cuda_compute(torch.tensor([data]))
#    
#    # Gather results from all processes
#    results = comm.gather(local_result, root=0)
#    
#    # Root process prints the results
#    if rank == 0:
#        return results
#        print("Results gathered from all processes:", results)


def pcall(f,xs):
    """Parallel call in the CPU with MPI"""
    from mpi4py import MPI
    
    # Initialize MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the process rank
    size = comm.Get_size()  # Get the total number of processes
    
    # Function to be parallelized
    def compute(data):
        # Example computation, replace with your own logic
        return f(data)
    
    # Data to be divided among processes
    if rank == 0:
        # Only the root process prepares the data
        chunks = [xs[i::size] for i in range(size)]
    else:
        chunks = None

    # Scatter the chunks to different processes
    chunk = comm.scatter(chunks, root=0)

    # Each process computes its part of the data
    local_result = [compute(item) for item in chunk]

    # Gather the results from all processes
    results = comm.gather(local_result, root=0)

    # Only the root process will return the result
    if rank == 0:
        flat_results = [item for sublist in results for item in sublist]
        return flat_results

# alias
pcallcpu = pcall

def check_mpi():
    try:
        from mpi4py import MPI
    except:
        return False
    # Initialize MPI communicator
    comm = MPI.COMM_WORLD
    # Get the total number of processes
    size = comm.Get_size()
    # Check if it is running with MPI
    if size > 1: return True
    else: return False



# now a simple test just to see if it works

if __name__=="__main__": # test if it is main
    def f(x):
        import time
        time.sleep(1)
        return x
    print(pcallcpu(f,[0,1,2,3,4]))
