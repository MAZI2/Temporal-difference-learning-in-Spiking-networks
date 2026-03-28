# SNN-Natural-Aversive-Behaviour

## Setup
### Clone NEST
```
git clone https://github.com/nest/nest-simulator.git
mkdir nest-build
cd nest-build
```

### Install cython
If using Conda:
```bash
conda install cython
```

### Linux:
```bash
cmake ../nest-simulator \
      -DCMAKE_INSTALL_PREFIX=../nest-install \
      -Dwith-python=ON \
      -DPYTHON_EXECUTABLE=$(which python3)
```

### Mac:
```bash
cmake ../nest-simulator \
  -DCMAKE_INSTALL_PREFIX=../nest-install \
  -Dwith-python=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -Dwith-openmp=OFF
```

### Build/Install
```bash
make -j2 or make -j (less stable)
make install
```

### Export paths

```bash
export PYTHONPATH=/path/to/nest-install/lib/python3.11/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/nest-install/lib:$LD_LIBRARY_PATH
```

### Build/Install NEST module

```bash
cd delayed_eligibility_synapse
mkdir build & cd build
cmake -Dwith-nest=/path/to/nest-install/bin/nest-config ..
make -j2
cd ..
./install.sh
```

On Linux you may need to run this before `./install.sh`:

```bash
source /path/to/nest-install/bin/nest_vars.sh
```
