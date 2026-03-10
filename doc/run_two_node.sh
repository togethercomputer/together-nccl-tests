export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun \
  -np 16 \
  -N 8 \
  -H pg24a-2-1-hpc.cloud.together.ai:8,pg24a-1-4-hpc.cloud.together.ai:8 \
  --bind-to none \
  -mca pml ob1 \
  -mca btl tcp,self \
  -mca btl_tcp_if_include eno16995np0 \
  -x LD_LIBRARY_PATH \
  -x NCCL_DEBUG=WARN \
  -x NCCL_IB_DISABLE=1 \
  -x NCCL_SOCKET_IFNAME=eno16995np0 \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
