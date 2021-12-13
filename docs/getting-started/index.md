# Access to machines

A step-by-step guide on how to access GPU equipped machines at CERN, CMS or how to develop on your machine.

## Prerequisites

1. CERN computing account

## Access machines at CERN

See the [CERN cloud insfrastructure resources guide](https://clouddocs.web.cern.ch/gpu/README.html) on how to request GPU resources.

1. lxplus

The lxplus service offers `lxplus-gpu.cern.ch` for shared GPU instances - with limited isolation and performance.

One can connect similary as would do to the `lxplus.cern.ch` host domain.

    ssh <username>@lxplus-gpu.cern.ch [-X]

## Access machines at CMS P5

This section is taken from the [CMS TWiki TriggerDevelopmentWithGPUs](https://twiki.cern.ch/twiki/bin/viewauth/CMS/TriggerDevelopmentWithGPUs) page.

### Dedicated machines for the development of the online reconstruction

There are 6 machines available for general development and validation of the online reconstruction on GPUs:

* **`gpu-c2a02-37-03.cms`**
* **`gpu-c2a02-37-04.cms`**
* **`gpu-c2a02-39-01.cms`**
* **`gpu-c2a02-39-02.cms`**
* **`gpu-c2a02-39-03.cms`**
* **`gpu-c2a02-39-04.cms`**

### All machines are equipped with

* two [Intel "Skylake" Xeon Gold 6130![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://ark.intel.com/content/www/us/en/ark/products/120492/intel-xeon-gold-6130-processor-22m-cache-2-10-ghz.html) processors (for a total of 2x16=32 physical cores and 2x2x16 = 64 logical cores or hardware threads);
* 96 GB of RAM;
* one [NVIDIA Tesla T4![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPU.

### How to connect

To connect to these machines you need to have an online account and be in the **`gpudev`** group.

To request access, please subscribe to the [cms-hlt-gpu@cern.ch](mailto:cms-hlt-gpu@cern.ch) e-group and send an email to [andrea.bocci@cern.ch](mailto:andrea.bocci@cern.ch), indicating

* whether you already have an online account;
* your online or lxplus username;
* your full name and email.

## Miscellaneous - or special GPU nodes

This section is more or less taken from the [Patatrack website systems](https://patatrack.web.cern.ch/patatrack/private/systems/cmg-gpu1080.html) subpage.

### cmg-gpu1080

#### System information

[Topology of the machine](https://fpantale.web.cern.ch/fpantale/out.pdf)

#### Getting access to the machine

In order to get access to the machine you should send a request to subscribe to the CERN e-group: cms-gpu You should also send an email to [Felice Pantaleo](mailto:felice.pantaleo@cern.ch) motivating the reason for the requested access.

#### Usage Policy

Normally, no more than 1 GPU per users should be used. To limit visible devices use

    export CUDA_VISIBLE_DEVICES=<list of numbers>

Where `<list of numbers>` can be e.g. `0`, `0,4`, `1,2,3`. Use `nvidia-smi` to check available resources.

#### Usage for ML studies

If you need to use the machine for training DNNs you could accidentally occupy all the GPUs, making them unavailable for other users.

For this reason you're kindly asked to use

`import setGPU`

before any import that will use a GPU (e.g. tensorflow). This will assign to you the least loaded GPU on the system.

It is strictly forbidden to use GPUs from within your jupyter notebook. Please export your notebook to a python program and execute it. The access to the machine will be revoked when failing to comply to this rule.