import numpy as np

# this function needs to be properly tested

def pcall(fun,data):
    """Call a function in parallel using MPI"""
    from mpi4py import MPI
    from ..parallel import cores
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    size = cores # how many cores
    
    # Master process distributes work
    if rank == 0:
        # Split data into chunks for each process
        chunk_size = len(data) // size
        remainder = len(data) % size
        
        chunks = []
        start = 0
        for i in range(size):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(data[start:end])
            start = end
        
        # Send chunks to worker processes
        for i in range(1, size):
            comm.send(chunks[i], dest=i, tag=11)
        
        # Master processes its own chunk
        my_chunk = chunks[0]
        
    else:
        # Worker processes receive their chunk
        my_chunk = comm.recv(source=0, tag=11)
    
    # Each process computes on its chunk
    results = []
    for value in my_chunk:
        result = fun(value)
        results.append((value, result))
    
    # Gather all results at master
    all_results = comm.gather(results, root=0)
    
    # Master process combines results
    if rank == 0:
        combined_results = []
        for proc_results in all_results:
            combined_results.extend(proc_results)
        
        # Sort by original value
        combined_results.sort(key=lambda x: x[0])
    return combined_results
        


