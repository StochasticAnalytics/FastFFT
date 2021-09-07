# <span style="color: #e87502">***Tutorials***</span>

Template matching intro TODO:

## 1 - 2DTM using the GUI


---
**Prerequisite** - 2DTM requires an alpha version of *cis*TEM. If you have not done so, please start by [installing one](../../sim/tutorials/get_cistem.md), and return here when finished.

### Project Setup
---


#### Download the EM data

We first need to obtain the data for the tutorial. If you have gone through the *cis*TEM single particle tutorial, then you already have it on hand. If not, we will be processing a single movie from that data set, which may be obtained from the CLI:

```bash
mkdir ${HOME}/cistem_tm_tutorial_1 ${HOME}/cistem_tm_tutorial_1/movies
cd ${HOME}/cistem_tm_tutorial_1/movies
wget -m ftp://ftp.ebi.ac.uk/empiar/world_availability/10146/data/May08_03.05.02.bin.mrc

# The full file structure is also created, so to save us some clicking, we link to the downloaded movie. Note, this is not necessary.
ln -s ftp.ebi.ac.uk/empiar/world_availability/10146/data/May08_03.05.02.bin.mrc  movie.mrc
```
#### Download the PDB model

Because these movies are binned, they are not particularly well compressed, so while you wait for your download to finish, we'll also grab the [PDB for horse spleen apoferritin](https://www.rcsb.org/structure/2w0o) that we'll use to generate our template. 

```{note}
You'll need to download the PDB format. Support for PDBx/mmCIF is brewing, but not yet available.
```
```{tip}
Some pdb files only include coordinates for the asymmetric unit, like 2w0o.pdb apoferritin. When you click to download, select the "Biological Assembly" to get a PDB with all the atoms specified.
```

### Pre-process the data
---
#### **create project**

* open *cis*TEM 
```bash
${HOME}/cisTEM_alpha/src/cisTEM
```
* create a new project in the cistem_tm_tutorial_1 folder
% TODO: this has a white box to cover up the "beta" logo, the white is a bit off. Replace when available.
![create project](../../../icons/TM_tutorials/tutorial_1/TM_tutorial_1_pic1.svg)

#### **import movie asset**

Different types of data in *cis*TEM are referred to as assets. Import your movie assest, with data parameters:

```{margin}  <span style="color: purple">*Discussion*:</span>
These images were collected with a spherical aberration corrector [see discussion TODO].
```

* Voltage: $300 ~keV$
* Spherical Aberration: $0.001~ mm$
* Pixel Size: $1.5 ~\mathring{A}$
* Exposure Per Frame: $2.0~ e^{-}/\mathring{A}^2$



![create project](../../../icons/TM_tutorials/tutorial_1/TM_tutorial_1_pic2.svg)

#### **align your movie**
% TODO: fix math text 
```{tip}
We tend to see the best detection with a total exposure between 30-50 $e^{-}/\mathring{A}^2$. When aligning your movies, you may select a range of frames to average together, we we choose 20 frames, corresponding to 40 $e^{-}/\mathring{A}^2$.
```
![create project](../../../icons/TM_tutorials/tutorial_1/TM_tutorial_1_pic3.svg)

#### **measure your ctf**
---
You can use the default settings for this step. Your results should be similar to below:

![create project](../../../icons/TM_tutorials/tutorial_1/TM_tutorial_1_pic4.svg)

### Generate your template
---
Instructions for this step can be found in [simulation tutorial 2.](calc_3d_scattering)

### Import your template
---
1) Use the left toolbar to open the assets panel
2) Select 3D volumes
3) Select import
4) Select add files, and choose the template you generated in the previous step
5) Enter 1.5 for the pixel size, which will enable the "Import" button

![create project](../../../icons/TM_tutorials/tutorial_1/TM_tutorial_1_pic5.svg)

### Setup run profile
---

```{margin}  <span style="color: purple">*How-to*:</span>
We let cisTEM pick the GPU with the most available memory. For better control on systems with multiple gpus, see this how-to page TODO:
```

1) Use the left toolbar to open the settings panel
2) Duplicate your default run profile, then click rename and type in gpu_1
3) 2x click the run profile entry to edit
4) Change the number of threads to 4
5) Be sure to click save!


![create project](../../../icons/TM_tutorials/tutorial_1/TM_tutorial_1_pic6.svg)

```{tip}
On some screens, this edit run profile dialog doesn't automaticallyexpand the whole way to expose the "ok" and "cancel" buttons. As a workaround, expand to full screen with the square icon.
```

### Search

1) Use the left toolbar to open the experimental panel
2) Change the symmetry to octahedral (o) in the dropdown, and togel the defocus seasrch off
3) Select our gpu_1 run profile
4) 2x check that the Use GPU radio button is selected
5) Start your search!

![create project](../../../icons/TM_tutorials/tutorial_1/TM_tutorial_1_pic7.svg)

```{note}
With the symmetry, no defocus search, and small image size, this search should only take 2-3 minutes depending on your GPU.
```
### Analyze results
1) Select the MT Results icon
2) Check the number of statistically significant detections
3) Check to see how well the search conformed to gaussian statistics
4) Scroll through the peak list, which highlights detections in your ...
5) image display options
  a) image
  b) scaled mip
  c) plotted result


![create project](../../../icons/TM_tutorials/tutorial_1/TM_tutorial_1_pic8.svg)

## 2 - 2DTM using the CLI
---
% TODO:
TODO

## 3 - Refine 2DTM results GUI
---
% TODO:
TODO

## 4 - Refine 2DTM results CLI
---
% TODO:
TODO
## End matter

If you've made it through these tutorials, you are ready for some more advanced usage. Please have a look over the discussion into link TODO for an overview of how you might design an experiment where you use 2DTM.

Once you are better oriented to the theory, please have a look at the How-to guides which give recipes for more specific scenerios, with relevant links to detailed discussion and reference material.

Get matchin'!


