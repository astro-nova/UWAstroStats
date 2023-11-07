# Intro

In this project, you will use machine learning and MCMC to estimate mass and redshift of galaxies using their photometry. 

This is usually done with SED fitting tools, such as [Prospector](https://github.com/bd-j/prospector). 

Prospector takes total flux measurements in a number of photometric bands (purple points on the animation) and the shape of each band as inputs (purple curves), and fits a spectrum to this data (in grey).

![Prospector SED fitting](https://github.com/bd-j/prospector/raw/main/doc/images/animation.gif)

With just a few points, e.g. optical *i* band only, the fit is really uncertain, because we don't know anything about what the young stars or dust or AGN are doing in our galaxy! As you add more points, the fit becomes better constrained. It's best to sample the full spectrum, from UV to IR, to constrain the behaviour of young and old stars in blue/optical as well as dust and AGN in the IR.

## Data

We will use the [Chang et al.](https://irfu.cea.fr/Pisp/yu-yen.chang/sw.html) catalog of ~850,000 galaxies that have flux measurements in the optical with SDSS *ugriz* bands, in the IR with WISE, and in the UV with GALEX. This gives us 11 bands to fit simultaneously.

The authors have already performed SED fitting using the [MAGPHYS](https://www.iap.fr/magphys/) code, except they used known spectroscopic redshifts from SDSS.

In this project, you will use the tools you've learnt to estimate masses and redshifts from photometry alone, and see how much your results agree with the MAGPHYS estimates.