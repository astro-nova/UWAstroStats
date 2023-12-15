# Things I did

## Set-up

1. Clone prospector and run `bash conda_install.sh`
    - Best not to do this inside your git repo because it's a large package and we don't want to push/pull it. 
    - Make sure to follow the directions from the `conda_install.sh` output: add `SPS_HOME` to your `~/.bashrc`, restart with `bash` to make changes take effect.
    - This creates a new environment, `prospector`, that has prospector and all its dependancies installed. Activate it with `conda activate prospector`. You will want to use other packages during this course, so make sure to intall them as well in the prospector environment.







## Required packages
We should make requirements.txt
- Your preferred between `jupyterlab` or `jupyter` to run our interactive notebooks
- `astroquery` (needed by the prospector tutorial not us so maybe not)
