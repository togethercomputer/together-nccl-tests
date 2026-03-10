# Two-Node NCCL Test Setup Guide

**Environment:**
- Node 1: `pg24a-2-1-hpc.cloud.together.ai`
- Node 2: `pg24a-1-4-hpc.cloud.together.ai`
- GPUs: 16x NVIDIA B300 SXM6 AC (8 per node)
- NCCL: 2.29.7+cuda13.2
- OpenMPI: 4.1.9a1

---

## Step 1: Install NCCL

Run on **both nodes**:

```bash
sudo apt-get install -y libnccl2 libnccl-dev
```

---

## Step 2: Build nccl-tests with MPI

Run on **both nodes**:

```bash
cd ~/nccl-tests
make clean
make MPI=1 MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
```

---

## Step 3: Configure SSH Passwordless Access

Run on **both nodes** — generate SSH keypair:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

Add local public key to own `authorized_keys` (both nodes):

```bash
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
```

Exchange public keys between nodes manually:
- Copy `~/.ssh/id_ed25519.pub` from Node 1 → append to `~/.ssh/authorized_keys` on Node 2
- Copy `~/.ssh/id_ed25519.pub` from Node 2 → append to `~/.ssh/authorized_keys` on Node 1

Verify SSH connectivity from Node 1:

```bash
ssh pg24a-1-4-hpc.cloud.together.ai hostname
```

---

## Step 4: Configure SSH ProxyCommand for MPI

MPI uses its own SSH invocation and requires the proxy to be set in `~/.ssh/config`.

Create `~/.ssh/config` on **Node 1 (launch node)**:

```bash
cat > ~/.ssh/config << 'EOF'
Host pg24a-*
    StrictHostKeyChecking no
    ProxyCommand /usr/bin/sss_ssh_knownhostsproxy -p 22 %h
EOF
chmod 600 ~/.ssh/config
```

Add both nodes to `known_hosts`:

```bash
ssh-keyscan -H pg24a-2-1-hpc.cloud.together.ai pg24a-1-4-hpc.cloud.together.ai >> ~/.ssh/known_hosts
```

Verify MPI can reach both nodes:

```bash
export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun \
  -np 2 \
  -H pg24a-2-1-hpc.cloud.together.ai:1,pg24a-1-4-hpc.cloud.together.ai:1 \
  --bind-to none \
  -x LD_LIBRARY_PATH \
  hostname
```

---

## Step 5: Run NCCL Test (TCP Mode)

Run on **Node 1 only**:

```bash
bash ~/nccl-tests/t2.sh
```

Contents of `t2.sh`:

```bash
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
```

---

## Step 6: IB Diagnostics (for future use)

Check IB device status:

```bash
ibstat
ibv_devinfo | grep -E "hca_id|state|rate|link_layer|fw_ver"
show_gids
```

Check IB partition keys:

```bash
cat /sys/class/infiniband/mlx5_0/ports/1/pkeys/* | grep -v "^0x0000$"
```

Test raw IB bandwidth between nodes:

```bash
# On Node 2 (server):
ib_write_bw -d mlx5_0

# On Node 1 (client):
ib_write_bw -d mlx5_0 pg24a-1-4-hpc.cloud.together.ai
```

Check NVIDIA kernel module options:

```bash
cat /proc/driver/nvidia/params | grep -E "EnableStreamMemOPs|PeerMapping|RegistryDwords"
```

Check NVLink topology:

```bash
nvidia-smi nvlink -s
nvidia-smi topo -m
```

---

## Key Flag Reference

### mpirun flags

| Flag | Description |
|------|-------------|
| `-np 16` | 16 total MPI ranks (1 per GPU) |
| `-N 8` | 8 ranks per node |
| `-H node1:8,node2:8` | Node list with slots |
| `--bind-to none` | Don't bind ranks to CPUs |
| `-mca pml ob1` | Use ob1 PML instead of UCX |
| `-mca btl tcp,self` | Use TCP transport (bypass IB) |
| `-mca btl_tcp_if_include eno16995np0` | Use specific network interface |
| `-x VAR` | Propagate env variable to all ranks |

### nccl-tests flags

| Flag | Description |
|------|-------------|
| `-b 8` | Min message size: 8 bytes |
| `-e 8G` | Max message size: 8 GB |
| `-f 2` | Step factor (doubling) |
| `-g 1` | 1 GPU per MPI rank |

### NCCL env variables

| Variable | Value | Description |
|----------|-------|-------------|
| `NCCL_IB_DISABLE` | `1` | Disable IB, use socket transport |
| `NCCL_SOCKET_IFNAME` | `eno16995np0` | Network interface for NCCL sockets |
| `NCCL_IB_HCA` | `mlx5_0,...` | Specify IB HCAs when IB is enabled |
| `NCCL_IB_GID_INDEX` | `0` | GID index for IB devices |
| `NCCL_DEBUG` | `WARN/INFO` | NCCL log verbosity |
| `NCCL_NET_GDR_LEVEL` | `5` | GPU Direct RDMA level |

---

## Known Issues

### IB RDMA Not Working Between Nodes

- **Symptom:** `ib_write_bw` fails with `IBV_WC_RETRY_EXC_ERR` (syndrome 0x81)
- **Hardware status:** 8x NDR 800G IB ports per node, all Active, same SM (lid 1), same pkey (`0x7fff`)
- **Root cause:** SM routing table has no path between LID 1958 (Node 1) and LID 1773 (Node 2)
- **Action required:** Infra team to enable IB routing between the two nodes on the NDR fabric

### Performance Impact

| Transport | Bus Bandwidth |
|-----------|--------------|
| TCP (current) | ~7.7 GB/s |
| NDR 800G IB (expected after fix) | ~350–400 GB/s |
