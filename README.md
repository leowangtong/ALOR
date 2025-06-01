<div align="center">
<h1>Active Learning via Vision-Language Model<br>Adaptation with Open Data</h1>
</div>

In this work, we propose leveraging VLM's pretraining data by retrieving samples closely related to the downstream task, using them to augment the task-specific data for AL. As expected, incorporating this data into existing AL methods leads to significant performance improvements.
Given that our method exploits open-source VLM and open data, we refer to it as ${Active Learning with Open Resources (ALOR)}$.

Further analysis of retrieved data reveals a naturally imbalanced distribution of task-relevant classes, exposing inherent biases within the VLM. This insight motivates our novel ${Tail First Sampling (TFS)}$ strategy for AL, an embarrassingly simple yet effective method that prioritizes sampling data from underrepresented classes to label.


![teaser](ALOR-flowchart_v10.png)



## Usage

### Installation
 
``` bash
conda create -n TFS python=3.10.14 
conda activate TFS 
pip install -r requirements.txt 
```
Install the datasets following the instructions in [DATASETS.md](DATASETS.md). 

See [RETRIEVAL.md](https://github.com/tian1327/SWAT/blob/master/retrieval/RETRIEVAL.md) for step-by-step instructions for retrieved data.

### Running ALOR

You can run ALOR that assembles CT, RDA and TFS using the following bash scripts.

```bash
# Usage: bash scripts/alvlm/run_train_lp_ft_ct.sh <dataset> [seed] <ALMETHOD> <round>

# ALOR: example for Round-0 active selection using TFS
bash scripts/alvlm/run_train_lp_ft_ct.sh semi_aves 666 TFS 0
```

- **dataset** $\in$ [semi_aves, food101, aircraft, oxford_pets, stanford_cars]
- **seed**: integer 
- **ALMETHOD** $\in$ [TFS, badge, coreset, entropy, logo, badge_pcb, alfa_mix]
- **round**: integer

The results of the experiments will be saved in the `output` directory.

### Running other adaptation methods
Below we provide the commands to run other adaptation methods.

 Update the `methods` option in the `scripts/alvlm/run_train_lp_ft_ct.sh` to use different adaptation methods (LP, FT, CT). And you can run PT the following bash scripts.
```bash
sh scripts/alvlm/run_train_pt.sh semi_aves open_vit_b32 TFS 666 none
```

## Acknowledgment
This code base is developed with some references on the following projects. We sincerely thank the authors for open-sourcing their projects.

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [SWAT](https://github.com/tian1327/SWAT)
- [PCB](https://github.com/kaist-dmlab/pcb)

