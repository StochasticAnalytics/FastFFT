# <span style="color: darkred">***Tutorials***</span>


(seeAbove1)= 
## 1 - Get an alpha version of *cis*TEM

To run the simulator or template matching you will need an alpha version of *cis*TEM. For most people, the best choice will be to download a pre-compiled binary, following the instuctions here: [**get cistem tutorial**](get_cistem.md) For those interested in compiling form source code, you will need to add the "--enable-experimental" flag to your configure line, as well as following the instructions [here.](https://github.com/bHimes/cisTEM_downstream_bah/wiki/Compiling-cisTEM)

(calc_3d_scattering)=
## 2 - Calculate a 3D scattering potential

Simulation in TEM involves describing

1) The Coulomb potential of the sample
2) The relativistic wave function of the imaging electrons
3) The interaction of 1 & 2
4) The influence of the microscope optics, including the lenses, detectors etc.

Step one is also the first step in a template matching experiment, were the detection efficiency is directly tied to the output SNR from the matched filter used in 2DTM. The SNR in turn depends strongly on how well your calculated specimen potential matches the imaged specimen potential recorded in your micrographs(s). To improve detection, both points 1 & 4 from above need to be carefully handled. 


### Materials needed
* An alpha version of *cis*TEM [see above.](seeAbove1)
* Information about the imaging conditions used in the data you wish to search.
* A PDB representing your molecule
* You may [**adapt this script**](../../TM/tutorials/make_3d_ref.md) to your specific use case. 

```{warning}
Only classic [PDB format](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) is supported at the moment. Support for newer PDBx/mmCIF is planned, but in the interm, you will need to manually convert to PDB using a tool like Chimera for example. Simply open your mmCIF and then "save as PDB."
```
```{tip}
Some pdb files only include coordinates for the asymmetric unit, like 2w0o.pdb apoferritin. When you click to download, select the "Biological Assembly" to get a PDB with all the atoms specified.
```

## 3 - Calculate a stack of noisy particles

### Materials needed
{{testVal}}

* An alpha version of *cis*TEM [see above.](seeAbove1)
```{margin}
Tip: Hover over the right side of the code box and click the clipboard icon to copy the link.
```
* The beta galactosidase PDB and a run script 
```bash
wget https://github.com/bHimes/cisTEM_docs/raw/main/tutorial_materials/bgal_flat.pdb https://github.com/bHimes/cisTEM_docs/raw/main/tutorial_materials/bgal_flat.sh
```
* 

### Overview

In this tutorial, you will use the script you downloaded to understand the fundamentals of what is produced in the image simulation, and the primary sources of shot noise and structural noise. 


### A) Simulate an isolated protein

**You will first run your bgal_flat.sh script as is. In subsequent steps we will modify parameters and command line options to change the results.**
```{tip}
Make sure that your script has the permissions to run after you download it. 
```bash
chmod u+x bgal_flat.sh
```

The first run is set to produce the projected Coulomb potential of the atoms in the enzyme betagalactosidase. Normally, when simulating a particle stack, we want to have a random set of orientations, however, to compare to the results in the following figures, we disable the randomization of the angles. ***Note*** that that what is produced is really the complex modulus of the detector wavefunction. This includes the abberations of the objective lens for standard cryoEM imaging, as shown in Figure 1A.

```bash
optional_args=" --save-detector-wavefunction --skip-random-angles "
```

You may want to adjust the number of threads used in your run to match your computer's hardware. Even 16 is probably overkill for these simple simulations.

```bash
output_filename="betgal_vacuum.mrc"
input_pdb_file="bgal_flat.pdb" # the parser is fairly good with PDB format, but will probably break in some cases.
output_size=-320 # pixels - if < 0 a hack to set a fixed size for 2d image simulation, if > 0 the size of a 3D reference volume to simulate.
n_threads=16 # Not so important for a a 3d, it will be fast either way.
```

```{figure} ../../../icons/SIM_tutorials/tutorial_2/Sim_tutorial2_1.svg
---
scale: 50%
align: center
---
**A) simulation ***in vacuuo*** B) adding pseudo-water molecules C) low-dose with  $1.0~ e^{-}/\mathring{A}^2$ D) low-dose with  $30.0~ e^{-}/\mathring{A}^2$ E) same as (D) farther from focus ($2.4\mu{m}$)**
```

Of course, proteins do not exist in a vacuum, they are embedded in solvent. We can include the solvent, by resetting the scaling factor to its default value of 1.0. Doing this, you should get an image that looks like Figure 1B.

```bash
output_filename="betgal_water.mrc"
.
water_scaling=1.0 # Linear scaling of water molecules - normally not used, except for demonstration purposes
```

```{warning}
Be careful to modify the parameters specified in each code block before executing the run script. In practice, you don't need the script, and can run the program interactively.
```

It is important to realize that this is still a "perfect" image, even though it is far noisier than that in (A). The additional signal from the solvent is one form of ***structural noise.*** Fortunately for us, this image is not actually a "perfect" image, because the water molecules move much more than the protein, which somewhat improves the contrast between protein and solvent. [TODO link to discussion]. One of the primary sources of noise in cryoEM is due to the low-dose imaging conditions that lead to an uncertainty in how many electrons arrive at the detector in a given time interval. This is called shot noise, which we observe in Figure 1C by removing the *** --save-detector-wavefunction*** argument.

```bash
output_filename="betgal_1elec_per_angSq.mrc"
.
optional_args=" --skip-random-angles "
```

To get a more realistic picture, we can simulate a short movie, with a total of $30.0~ e^{-}/\mathring{A}^2$. The length of the timestep in each movie frame is set by the exposure per frame. You'll need to change the number of frames to be 20 in order to reproduce Figure 1D.

```bash
output_filename="betgal_30elec_per_angSq.mrc"
.
exposure_per_frame=1.5 # e-/Ang^2 ;; Dose rate doesn't affect 3D sim (only 2d) here it is used with n_frames to get the total exposure
exposure_rate=3.0 # e-/ pixel /s
n_frames=20
pre_exposure=0 # If you've left off some early frames, account for that here
```

Even with the longer exposure, this protein is still very hard to see. If you increase the defocus from **$8000\mathring{A}$** to **$24000\mathring{A}$**

```bash
output_filename="betgal_30elec_per_angSq_highdefocus.mrc"
.
wanted_defocus=240000 # Angstrom
```

**You should have now reproduced Figure 1, learning about shot noise and structural noise along the way. It is important to point out that the improved visibility of the protein by imaging farther from focus does not mean the image has more information, in fact it has less.**

---
### B) Simulate a more realistic protein

So far, we have been simulating a single lonely protein devoid of neighbors, in a minimally thick layer of solvent. The first thing we will do, is stop supressing the random angles and simulate a small particle stack so you can see what a more realistic output will look like. For visualization purposes, we will again downweight the water, this time by a factor of 0.5, and turn back on the ***--save-detector-wavefunction*** option. After changing these parameters, you should be able to reproduce Figure 2.

```bash
output_filename="betgal.mrc"
wanted_defocus=8000 # Angstrom
n_frames=1
water_scaling=0.5 # Linear scaling of water molecules - normally not used, except for demonstration purposes
optional_args=" --save-detector-wavefunction"

```

```{margin}
***NOTE*** your particles will be in a different orientation than mine. You also may need to adjust the global image contrast in your image display software, as each particle in this case may have a different solvent thickness.
```

```{figure} ../../../icons/SIM_tutorials/tutorial_2/Sim_tutorial2_2.svg
---
scale: 50%
align: center
---
**Random particles in solvent**
```

The default behavior is to specify a layer of solvent just big enough to hold your protein, so an oblong protein like betagal (**$~190\mathring{A}$** in the longest dimension) will produce variable contrast. You can set the minimum thickness of your ice layer, assuming it is at least as big as the default. Here we'll set the thickness to 200nm to illustrate how more solvent reduces contrast.
```bash
output_filename="betgal_thick.mrc"
minimum_thickness=2000 # Angstrom
```
```{margin}
Remember that the solvent is artificially multiplied by 0.5 in this tutorial example. Do you think you'll be able to see a ~450kDa protein in 200nm ice?
```

```{figure} ../../../icons/SIM_tutorials/tutorial_2/Sim_tutorial2_3.svg
---
scale: 50%
align: center
---
**Random particles in thick layer of solvent**
```

In the previous example we added more solvent molecules. Obviously, other protein molecules are another source of structural noise. You can add up to six random neighboring particles by including the ***--max_number_of_noise_particles=6*** command line flag. We'll also reduce the thickness of our solvent layer to 25nm.

```bash
output_filename="betgal_neighbors.mrc"
minimum_thickness=250 # Angstrom
optional_args=" --save-detector-wavefunction --max_number_of_noise_particles=6 "

```


```{figure} ../../../icons/SIM_tutorials/tutorial_2/Sim_tutorial2_4.svg
---
scale: 50%
align: center
---
**Particle stack with neighbors**
```

Finally, we'll put it all together, setting a higher defocus, moderate ice thickness, noise particles, full water intensity, and a movie with 45 electrons total exposure.

```bash
output_filename="betgal_realistic.mrc"
minimum_thickness=250 # Angstrom
water_scaling=1.0
wanted_defocus=12000 # Angstrom
n_frames=30
optional_args=" --max_number_of_noise_particles=6 "


```

```{figure} ../../../icons/SIM_tutorials/tutorial_2/Sim_tutorial2_5.svg
---
scale: 50%
align: center
---
**Realistic particle stack**
```
---
### Summary
TODO: List of key concepts and links to How-tos (e.g. modify the distribution of random noise particles) and discussions (e.g. limits on expsoure/frame, exposure rate etc.)


