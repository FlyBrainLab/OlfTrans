========
OlfTrans
========

Olfactory Transduction Utilities for FlyBrainLab.


* Free software: BSD license
* Documentation: https://flybrainlab.github.io/OlfTrans/

Features
--------

The main features of the package are 3 fold:

1. Estimating Binding and Dissociation Rates from Data (see next session for available datasets)
2. Running OTP-BSG Cascades
3. Working with other FlyBrainLab libraries


Datasets
--------

**OlfTrans** relies on the following datasets to estimate model parameters for
*Drosophila* adult and larva Odorant Transduction:

1. (Included) Hallem, E. A., & Carlson, J. R. (2006). *Coding of Odors by a Receptor Repertoire*. Cell. `DOI1 <https://doi.org/10.1016/j.cell.2006.01.050>`_
    - Accessible as :code:`olftrans.data.HallemCarlson`
2. (Included) Kreher, S. A., Kwon, J. Y., & Carlson, J. R. (2005). *The molecular basis of odor coding in the Drosophila larva*. Neuron. `DOI2 <https://doi.org/10.1016/j.neuron.2005.04.007>`_
    - Accessible as :code:`olftrans.data.Kreher`
3. (Need Download) Lazar, A. A., & Yeh, C.-H. (2020). *A molecular odorant transduction model and the complexity of spatio-temporal encoding in the Drosophila antenna*. PLOS Computational Biology. `DOI3 <https://doi.org/10.1371/journal.pcbi.1007751>`_
    - Electrophysiology data of *Drosophila* adult antenna can be downloaded `here <http://amacrine.ee.columbia.edu:15000/>`_
    - Rename the downlaod the file (77 MB) as :code:`antenna_data.h5` and place into the :code:`olftrans/data/` folder
    - Afterwards, accessible as :code:`olftrans.data.Physiology`



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

Many Utilities/Plotting functions are adapted from Neural_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Neural: https://github.com/chungheng/neural