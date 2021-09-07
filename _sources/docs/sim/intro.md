# <span style="color: darkred">**Intro**</span>


Image simulation plays a central role in the development and practice of high-resolution electron microscopy, including transmission electron microscopy of frozen-hydrated specimens (cryo-EM). Simulating images with contrast that matches the contrast observed in experimental images remains challenging, especially for amorphous samples. Current state-of-the-art simulators apply post hoc scaling to approximate empirical solvent contrast, attenuated image intensity due to specimen thickness, and amplitude contrast. This practice fails for images that require spatially variable scaling, e.g., simulations of a crowded or cellular environment. Modeling both the signal and the noise accurately is necessary to simulate images of biological specimens with contrast that is correct on an absolute scale. The “Frozen-Plasmon” method is introduced which explicitly models spatially variable inelastic scattering processes in cryo-EM specimens. This approach produces amplitude contrast that depends on the atomic composition of the specimen, reproduces the total inelastic mean free path as observed experimentally, and allows for the incorporation of radiation damage in the simulation. These improvements are quantified using the matched-filter concept to compare simulation and experiment. The Frozen-Plasmon method, in combination with a new mathematical formulation for accurately sampling the tabulated atomic scattering potentials onto a Cartesian grid, is implemented in the open-source software package cisTEM. 

## Recommended reading

* The primary source for this is our theory paper, which can be found [on bioarxiv.](https://www.biorxiv.org/content/10.1101/2021.02.19.431636v1)

