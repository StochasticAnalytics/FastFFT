# 3d reference script

The following script assums you have a working alpha version of cisTEM installed on your system, by following the instructions [here.](../../sim/tutorials/get_cistem.md) If you have installed *cis*TEM at some other location, you will need to modify the **path_to_script** variable below.

```{tip}
The matched filter is substantially more sensitive to having an accurately calibrated pixel size than your typical SPA refinement. You can refine your pixel size by using 2DTM, however, it is best practice to ensure it is as accurately calibrated as possible to start with.
```

```bash
#/bin/bash


# Specify where your alphas version of cisTEM is installed. 
path_to_cistem="${HOME}/cisTEM_alpha/src"

# Singulatiry can be run with "singularity shell" to give you an "interactive" env in the container.
# In this example, we instead execute a specific function in the container
function_to_execute="simulate"

# If you are running match_template, you need the --nv flag as below, if the flag is included and the machine has no gpus, it will not run.
# Singularity will bind your homedirectory, and should bind the PWD. Outside of this, it will not know about your filesystem
# If you want to give access (say your images are on nrs or a scratch disc somewhere, add the --bind directive.

#singularity exec --nv --bind /scratch_dublin/himesb /groups/himesb/notes/cisTEM_20210401 /cisTEM/bin/simulate

################################
# File things
################################
output_filename="2w0o_template.mrc"
input_pdb_file="2w0o.pdb" # the parser is fairly good with PDB format, but will probably break in some cases.
output_size=256 # pixels - if apix is < 0.5 simulated at this size, if > 0.5 < 1.5 padded internally by 2x and Fourier cropped, > 1.5 apix, padded 4x
n_threads=4 # Not so important for a a 3d, it will be fast either way.


################################
# Bluring things
################################
linear_scaling_of_PDB_bfactors=1.0 # make sure your model is reasonable i.e. the builder didn't set all bfactors to 50 or something. can be between 0 and 1
per_atom_bfactor=4.0 # add any bfactor to all atoms, may need to be increased since motion is not considered in the 3d simulation

################################
# Imaging things
################################
pixel_size=1.5
CS=0.001 # mm
KV=300 # kev
OA=100 # micron, objective aperture won't affect a 3D sim

exposure_per_frame=2 # e-/Ang^2 ;; Dose rate doesn't affect 3D sim (only 2d) here it is used with n_frames to get the total exposure
n_frames=20
pre_exposure=0 # If you've left off some early frames, account for that here


################################
# Don't worry about things
################################
# There are some options that I plan to set as default, but due to the way things are written, need to be passed via command line args.
from_future_args=" --only-modify-signal-3d=2 --wgt=0.225 --water-shell-only"

# There are several interactive options that need to be specified that aren't related to 3D sim, I've just filled these in below


################################
# Run the thing
################################

${path_to_cistem}/${function_to_execute} ${from_future_args} << EOF
$output_filename
$output_size
$n_threads
$input_pdb_file
no
$pixel_size
$linear_scaling_of_PDB_bfactors
$per_atom_bfactor
$exposure_per_frame
$n_frames
yes
no
1
0.0
$KV
$OA
$CS
0.0
$pre_exposure
32
10
2048
0.0
2.0
0.1
0.0001
0.0
0.0
0.0
0.0
EOF
```