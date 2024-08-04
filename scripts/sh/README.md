# Slurm@ICS

Students with an ICS account have access to the openlab Linux cluster on campus. The openlab cluster is a partition called `openlab.p`, and none of the nodes have a GPU that we can use for model training. Slurm allows students to submit jobs to other partitions that are GPU capable, like `opengpu.p`. <b>DISCLAIMER: Your activity on openlab is logged. If you abuse compute resources for personal purposes, the staff is not prepared to help you recover access to the openlab cluster</b>.

## How To Use Slurm
Submitting jobs with Slurm is incredibly simple and quick via the command line.
1. Login to openlab (for help, see [this](https://gist.github.com/ChaseC99/9506cf219ca33c60693ea4c4396a4c19))
2. In `scripts/train.py` add these lines:
    ```python
    # With your imports
    from dotenv import dotenv_values

    # In the function def train(), before initializing the wandb_logger
    # Set your API Key from .env using dotenv::dotenv_values
    config_wbapi = dotenv_values(".env")
    kv_api = config_wbapi["WB_API_KEY"]

    # Call wandb::login to bypass interactive terminal login
    # Required for teams submitting batch jobs to slurm
    wandb.login(key=kv_api, relogin=True)
    ```
2. Load the module with `module load slurm`
3. Submit a batch script to Slurm with `sbatch scripts/sh/<your_shell_script>.sh`

And... you're done! Close the terminal, shut down your computer, high-five your teammates. The opengpu partition will run it independent of your remote session on openlab until it finishes, timeouts, or crashes (more on that later). 

## How to configure the script(s)
Template shell script files have been created by staff. Fields that you need to complete have been marked with `#TODO`. In order to protect your private API key on Weights and Biases, you will need to create a `.env` file on the top level of the repository. The file itself only needs 2 lines of code (but you can add more to customize your project settings):
```python
# .env:

    WB_API_KEY = <your_key> # Wanb.ai > User Settings > Danger Zone > API keys
    WB_UNAME = <your_wandb_username> # Wanb.ai > User Settings > Profile
```
- These values are used in the shell script and `train.py`, so they would be compromised if you tried to hard code them in your python source files!

Also be sure to create a directory called `slurm_files` in `scripts/sh/` since the path has been preconfigured to create the output and error files there.

## FAQ
### Do I need to use Slurm?
No, using Slurm is not required for the course. However, learning how to write shell scripts is an important and attractive skill for any engineer that wants to work in a field with any sort of job automation. It will also make your life easier.

### Why should I be interested in learning Slurm?
Using Slurm@ICS will take a little work, but it will dramatically increase the flexibility of model training. Some main issues that come up if you use your own hardware:
- Model training is expensive. Your computer is going to be drawing power continuously until you finish. Even the free tier of Google Collab requires you to keep your session open in the web browser.
- Model training can take days. Do you know, right now, if you can start training without your computer or laptop turning off, or even going to sleep so you can bring it to school for a week? If you use Slurm, you can close your IDE, turn off your computer, and Slurm will send you an email when the job is finished.
- Anyone on your team can submit jobs. Nobody has to be the dedicated hardware workhorse. The best way to maintain your accountability guideline is to remove the need to enforce them (foisting all the responsibility on a single person is a great way to run into accountability issues).

### Okay, I'm sold, but the training loop failed. Now what?
Since the job is run on another partition, the error logs will not be in your terminal. Instead, an error file will be generated in `scripts/sh/slurm_files/`. Common mistakes that would result in you not having these files is if you either didn't create the `slurm_files/` directory, or if you removed the `#` before `SBATCH` in the shell script file. These are not comments! They define the SBATCH lines as bash commands. 
If you are having further issues with your shell script, there are a lot of resources online. Consult these, in order, to the best of your ability before reaching out to staff for help:
1. [Slurm@ICS Documentation (VPN Required)](https://wiki.ics.uci.edu/doku.php/services:slurm)
2. [Slurm API Reference](https://slurm.schedmd.com/slurm.html)

### It works! Wait, why is it taking so long?
The opengpu partition, while relatively unfamiliar to most undergraduate students, is still available to anyone with an ICS account. Your batch script is placed in a queue to wait, then run, then get paused, then wait, run... you get the idea. There are multiple factors in job priority, so it will not be as simple as "volume of work * rate of progress = job completion time".

<b>Do not wait until the last minute to start training.</b> 

If time is of the absolute essence, you'll have to look into alternatives to train your model, since staff cannot grant your slurm job priority over others. [More on Slurm's Fair Tree Fairshare Algorithm](https://slurm.schedmd.com/fair_tree.html).

## Last Bit of Advice
<b>Don't spam jobs!</b> Gradescope and GitHub are forgiving to an extent, but your behavior on Slurm affects your resource allocations. Ideally you should be implementing your own unit tests, but you should at least confirm that you can get training started locally (test for a single epoch, small batch size) before you request resources through Slurm.

<b>NOTE:</b> The repo (including the data) needs to exist on the openlab servers. Everyone with an account for openlab gets 256gb of storage, so storing the repo on openlab souldn't be a problem. 
