These five files work together:Matcher_vacupot, WF_dpl_vacupot_slim, classifier, vacupot_plotter, and builder.
The "matcher" is the file used to run all five codes. 
It calls the builder if the WAVECAR has not already been parseed and the true bloch states contsructed and saved as npz. The builder needs VaspBandUnfolding and PySBT to be installed as modules.
Once the true bloch states have been constructed across the components and the full system, The WF_dpl_vacupot_slim finds the vacuum potential and the fermi energy from the DOSCAR and shift the energies of all states such that the vacuum potentials are align. Dipole Corrections must be on in VASP for this to properly work.
Then the matcher takes the squared absolute value of inner products between the full system states and all the component states.
The matcher then saves the matches and all system state overlap information as txt files it passes to the plotter.
The plotter reads the txt match files and then plots the final figure, calling the classifier to determine the component shifts that are shown on hover tools in the final plot.
Full system state hover tools show the energy and idx of that state as well as the dE and ov of the best matching states from each component. Full system states are colored by their best matching component states. The vacuum potentials are thick black lines while the fermi energies are dashed red lines.
