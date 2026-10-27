# QLoRA batch training on DeltaAI

Submit from a login node; Slurm starts Python on the allocated GPU. No Jupyter server or open laptop is required. The script requests one GPU, 8 CPUs, 64 GB RAM and an eight-hour maximum on `ghx4`. This is a time limit, not a runtime estimate; the allocation ends when the program exits.

## Upload from your Mac

```bash
cd /Users/danny/Desktop/IT299
rsync -av bridge_qlora_deltaai.ipynb deltaai-login:~/IT299/
rsync -av scripts/ deltaai-login:~/IT299/scripts/
rsync -av jobs/ deltaai-login:~/IT299/jobs/
```

The remote project also needs the existing `data/processed/train.csv`, `validation.csv`, `manifest.json`, and `results/bridge_baseline/model_revision.json`.

## Submit from the DeltaAI login terminal

```bash
cd ~/IT299
ls -ld /work/nvme/bihk
mkdir -p /work/nvme/bihk/dmorsovillo/IT299 /work/nvme/bihk/dmorsovillo/hf_cache
mkdir -p logs
sbatch jobs/train_qlora.sbatch
```

The script uses your verified Slurm Account `bihk-dtai-gh` (the Account column, not the descriptive Project column). Storage code `bihk` is inferred from that account; confirm the directory exists and is writable using the commands above before submitting. Record the job ID printed by `sbatch`. No interactive allocation is needed. You can change the limit with `sbatch --time=04:00:00 jobs/train_qlora.sbatch`.

## Monitor

Replace `JOBID` with the submitted job ID:

```bash
squeue -u "$USER"
tail -f logs/qlora-JOBID.out
```

Ctrl-C stops `tail`, not the job. Errors are in `logs/qlora-JOBID.err`. After the job leaves the queue:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

Successful execution prints `QLoRA training and final validation completed.` Results are under `/work/nvme/bihk/dmorsovillo/IT299/results/bridge_qlora/RUN_ID/`. The notebook is the code source; its displayed outputs are not updated by batch execution. Do not submit the same training simultaneously through Jupyter.

## Interrupted runs

Recovery checkpoints save every 100 optimizer steps and at epoch boundaries. Validation and best-checkpoint selection remain at epoch boundaries. These disk checkpoints are separate from gradient checkpointing, which remains disabled. Trainer retains up to two checkpoints (normally best and latest). Keep these on DeltaAI; they include optimizer state and are larger than adapter weights alone.

List completed checkpoints:

```bash
find /work/nvme/bihk/dmorsovillo/IT299/checkpoints/bridge_qlora -name checkpoint_complete.json
```

Choose a checkpoint from the interrupted run, then submit explicitly:

```bash
sbatch jobs/train_qlora.sbatch --resume /work/nvme/bihk/dmorsovillo/IT299/checkpoints/bridge_qlora/RUN_ID/checkpoint-100
```

Use the actual run ID and checkpoint step, not the example above. Resume requires full optimizer, scheduler, RNG and adapter state, unchanged training configuration and data, and any referenced best checkpoint. It reuses that run's output directory; repeat validation at the same step replaces that step's predictions. Only checkpoints made by this updated notebook have completion markers. There is no automatic requeue. If terminated before the first checkpoint, start a new run. Slurm status is authoritative after termination; `run_config.json` may still say `training` if the process was killed.

## Local syntax check

```bash
python3 scripts/run_qlora_batch.py --check
bash -n jobs/train_qlora.sbatch
```

These checks do not load the model or verify GPU training.

Model downloads use `HF_HOME=/work/nvme/bihk/dmorsovillo/hf_cache`. Code and small Slurm logs remain in `~/IT299`. Previous caches and results are not moved or deleted. To pull new metrics to your Mac:

```bash
rsync -av deltaai-login:/work/nvme/bihk/dmorsovillo/IT299/results/bridge_qlora/ /Users/danny/Desktop/IT299/results/bridge_qlora/
```

## Training import preflight

After uploading updated scripts, run in the DeltaAI login terminal:

```bash
cd ~/IT299
module load python/miniforge3_pytorch/2.11.0
conda activate base
python scripts/run_qlora_batch.py --check-imports
```

This only imports libraries; it does not load a model or train. The launcher also
performs this check before each run. The notebook and launcher disable optional
Transformer Engine within their Python process because the shared module has
incompatible TE package versions. NF4/BF16 training uses bitsandbytes and SDPA;
shared installed packages are not modified. Restart an existing notebook kernel
before running the updated notebook.
