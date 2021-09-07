# *Download an Install*
---

## A zipped binary can be found here

% WARNING {{ SUBSTITUTION }} see _config.yaml
The simulator is not included in the current major release of *cis*TEM, so you'll need to either compile from source, following instructions here, or for many linux users, 
[this binary](https://drive.google.com/file/d/1BDQmN3quI-bnOYe23l9p_mb7jWFEGS1f/view?usp=sharing)  compiled on centos7 should work for you.

## Unpack the binary in your home dir

To make *cis*TEM programs available at ${HOME}/cisTEM_alpha/src/program_name simply run
% WARNING {{ cisTEM_1_0aabb63_20210505.zip }} see _config.yaml
```bash
unzip -d ${HOME}/cisTEM_alpha ${HOME}/Downloads/cisTEM_1_0aabb63_20210505.zip
```

```{note}
The naming convention is a sequential index, the first seven characters in the corresponding git commit hash, and then the date the binary was compiled.
```