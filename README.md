{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf100
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww20140\viewh9720\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # blendfitter\
\
[beta version]\
\
*blendfitter* is a collection of tools to analysze blended sources, including methods to extract information on RVs and bisectors directly from the CCFs of HARPS & Co. It makes use of the packages *emcee* (Markov Chain Monte Carlo sampling), *celerite* and *george* (Gaussian Process models).\
\
The code is freely available at https://github.com/MNGuenther/blendfitter under the MIT License. Feedback and contributions are very welcome.\
\
If you use *blendfitter* or parts of it in your work, please cite *G\'fcnther et al., 2018* (link). Please also include the following acknowledgement: "This work makes use of the *blendfitter* package (*G\'fcnther et al., 2018*), is a collection of tools to analysze blended sources, including methods to extract information on RVs and bisecotrs directly from the CCFs of HARPS & Co. It makes use of the packages *emcee* (Foreman-Mackey 2013), *celerite* (Foreman-Mackey 2017) and *george* (\\ref). This work makes further use of the *python* programming language (Rossum 1995) and the open-source *python* packages *numpy* (van der Walt, Colbert & Varoquaux 2011), *scipy* (Jones et al. 2001), *matplotlib* (Hunter 2007), *tqdm* (doi:10.5281/zenodo.1468033) and *seaborn* (https://seaborn.pydata.org/index.html)."\
\
\
## Table of contents\
1. How to use & examples\
References\
\
\
## 1. How to use & examples\
\
a) You have a directory full of HARPS CCF .fits files you want to re-analyze using a GP baseline to account for any systematics? Go no further than calling:\
\
    import blendfitter\
    blendfitter.analyse_CCFs(indir)\
\
b) You want to read out the relevant infos from the HARPS CCF .fits headers?\
\
    import blendfitter\
    BJD, RV, RV_err, Contrast, FWHM = blendfitter.ccf_io.extract_HARPS_data_from_fits(fname)\
\
\
## References\
\
 - Foreman-Mackey, D., Hogg, D. W., Lang, D. & Goodman, J. (2013), Publications of the Astronomical Society of the Pacific, 125, 306\
 - Foreman-Mackey, D., Agol, E., Ambikasaran, S. & Angus, R. (2017), The Astronomical Journal, 154, 220\
 - Hunter J. D., 2007, Comput. Sci. Eng., 9, 90\
 - Jones E. et al., 2001, SciPy: Open Source Scientific tools for Python. Available at: http://www.scipy.org/\
 - Rossum G., 1995, Technical Report, Python Reference Manual, Amsterdam, The Netherlands\
 - van der Walt S., Colbert S. C., Varoquaux G., 2011, Comput. Sci. Eng., 13, 22\
\
\
\
}