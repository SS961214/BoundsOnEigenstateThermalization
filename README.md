# BoundsOnEigenstateThermalization [![DOI](https://zenodo.org/badge/919339114.svg)](https://doi.org/10.5281/zenodo.14707989)

# Prerequisites
- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- [nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)

# Instructions

## Building the Docker image
This code provides a Dockerfile that can be used to build the environment where the code runs.
In the repository root directory (where the Dockerfile is located), execute the following commands:
```shell
docker build . --tag mbodyeth:latest
```
This will create a container image that contains all the necessary dependencies required to compile and run the code.

## Running a Docker container
Once you have builded the container image, you will then need to run a container from the image.
To access the code from inside the container and the numerical results from outside the container, the repository root directory should be mounted to the container.
You can run a container with the root directory mounted by executing the following command in the repository root directory:
```shell
docker run --rm -it --runtime=nvidia --gpus all --mount type=bind,src=./,dst=/root/work -u $(id -u $USER):$(id -g $USER) mbodyeth:latest zsh
```
Once the container is successfully run, you will get inside it.

## Compiling and running codes to obtain the numerical data
The commands required to compile and run the code are provided in the shell script named "main.sh", which is located in the repository root directory.
```shell
chmod u+x main.sh && ./main.sh
```
Since the computations for large system sizes require much time, the input parameters are restricted to relativelly small sizes.
You can modify the parameters "NMin", "NMax", "mMin", "mMax" to calculate for large system sizes and many-body operators.

## Raw numerical data
If you run "main.sh", the code generates the numerical data for the squared seminorm of the difference between energy eigenstates and the mirocanonical ensemble, i.e.,

```math
\left( \left|\left| |E_{\alpha}\rangle\!\langle E_{\alpha}| - \rho^{(\mathrm{mc})}_{\delta E}(E_{\alpha}) \right|\right|_{2}^{(\mathcal{A}^{[0,m]}) } \right)^2 = \displaystyle\sum_{\mu=1}^{\mathrm{dim}\, \mathcal{A}^{[0,m]}} \left| \mathrm{tr}\, \Lambda_{\mu}^{\dagger} \left( |E_{\alpha}\rangle\!\langle E_{\alpha}| - \rho^{(\mathrm{mc})}_{\delta E}(E_{\alpha}) \right) \right|^2,
```

where $|E_{\alpha}\rangle$ is an energy eigenstate, $\rho_{\delta E}^{(\mathrm{mc})}(E)$ is the microcanonical density operator in an energy shell centered at energy $E$ with width $2\delta E$, and $\{\Lambda_{\mu}\}_{\mu}$ is an orthonormal basis of the m-body operator space $\mathcal{A}^{[0,m]}$.
The bounds $\Lambda_{2}^{[0,m]}$ on the (diagonal) ETH measure can be obtained by taking the maximum of the seminorms within an energy window where one tests the ETH.
It bounds the ETH measure $\Lambda_{1}^{[0,m]}$ as $\Lambda_{2}^{[0,m]} \leq \Lambda_{1}^{[0,m]} \leq \sqrt{D}\, \Lambda_{2}^{[0,m]}$, where $D$ is the dimension of the Hilbert space.

In spin systems, the $m$-body operator space $\mathcal{A}^{[0,m]}$ can be decomposed into the direct sum of the *exactly* m-body operator spaces
```math
\mathcal{A}^{(m)} := \mathrm{span}\left\{ \sigma_{x_{1}}^{(p_{1})} \cdots \sigma_{x_{m}}^{(p_{1})} \mid 1\leq x_{1} < x_{2} < \cdots < x_{m} \leq L,\ \forall p_{1} = 1,2,3 \right\},
```
where $\sigma_{x}^{(p)} \ (p=1,2,3)$ is the Pauli operator acting on site $x$.
Therefore, the code calculates the squared seminorm for $\mathcal{A}^{(m)}$ instead of $\mathcal{A}^{[0,m]}$, i.e.,
```math
\left( \left|\left| |E_{\alpha}\rangle\!\langle E_{\alpha}| - \rho^{(\mathrm{mc})}_{\delta E}(E_{\alpha}) \right|\right|_{2}^{(\mathcal{A}^{(m)}) } \right)^2 = \displaystyle\sum_{\mu=1}^{\mathrm{dim}\, \mathcal{A}^{(m)}} \left| \mathrm{tr}\, \Lambda_{\mu}^{\dagger} \left( |E_{\alpha}\rangle\!\langle E_{\alpha}| - \rho^{(\mathrm{mc})}_{\delta E}(E_{\alpha}) \right) \right|^2.
```

For generic systems with local and few-body interactions, the resulting files are named like "./results/mBodyETH/PBC_TI/<*Ensemble*>/SampleNo<*M*>/SystemSize_L<*L*>_N<*N*>/mBody_quasiETHmeasureSq_dE<*shellWidthParam*>.txt".
Here, the parameter values are as follows:
- <*Ensemble*>: ShortRange_Spin, ShortRange_Boson_$\ell$local_$k$body, ShortRange_Fermion_$\ell$local_$k$body
- <*M*>: The identifier of the sample Hamiltonian
- <*L*>: The number of lattice sites in a 1D chain
- <*N*>: Particle number (This is ommitted for spin systems for which we always have $L=N$.)
- <*shellWidthParam*>: The parameter with which the width of the microcanonical shell is given by $\delta E = (\mathit{shellWidthParam})/L$.

The columns of the output files correspond to the parameter $m$, which ranges from $m=1$ to $m=N$, and the rows correspond to the energy eigenvalue $E_{\alpha}$.
