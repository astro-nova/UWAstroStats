import time, sys

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer


# --------------
# RUN_PARAMS
# When running as a script with argparsing, these are ignored.  Kept here for backwards compatibility.
# --------------

run_params = {'verbose': True,
              'debug': False,
              'outfile': 'demo_galphot',
              'output_pickles': False,
              # Optimization parameters
            #   'do_powell': False,
            #   'ftol': 0.5e-5, 'maxfev': 5000,
            #   'do_levenberg': True,
            #   'nmin': 10,
              # emcee fitting parameters
              'nwalkers': 128,
              'nburn': [16, 32, 64],
              'niter': 512,
            #   'interval': 0.25,
            #   'initial_disp': 0.1,
              # dynesty Fitter parameters
              'nested_bound': 'multi',  # bounding method
              'nested_sample': 'unif',  # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_target_n_effective': 10000,
              # Model parameters
              'add_neb': False,
              'add_duste': False,
              'var_redshift' : False,
              'spec_z' : 0.0,
              # SPS parameters
              'zcontinuous': 1,
              }


# --------------
# Observational Data
# --------------

# Here we are going to put together some filter names
galex = ['galex_FUV', 'galex_NUV']
spitzer = ['spitzer_irac_ch'+n for n in '1234']
bessell = ['bessell_'+n for n in 'UBVRI']
sdss = ['sdss_{0}0'.format(b) for b in 'ugriz']

# The first filter set is Johnson/Cousins, the second is SDSS. We will use a
# flag in the photometry table to tell us which set to use for each object
# (some were not in the SDSS footprint, and therefore have Johnson/Cousins
# photometry)
#
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
filtersets = (galex + bessell + spitzer,
              galex + sdss + spitzer)


def build_obs(objid, **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """
    # Writes your code here to read data.  Can use FITS, h5py, astropy.table,
    # sqlite, whatever.
    from astropy.io import fits
    from astropy.table import Table
    import pandas as pd
    import sedpy
    import prospect

    from prospect.utils.obsutils import fix_obs

    with fits.open('data/sw_input.fits') as f:
        df = Table(f[1].data).to_pandas()
        f.close()

        """Given an object, load in fluxes, convert them to nanomaggies, and create a dict used in Prospector."""
    gal_id = objid
    inp = {}
    
    # Get dataframe row for the object
    row = df.iloc[gal_id]
    inp['redshift'] = row.redshift

    # Load the filter response curves from sedpy
    bands = [f'sdss_{filt}0' for filt in 'ugriz'] + [f'wise_w{n}' for n in range(1,5)]
    filters = sedpy.observate.load_filters(bands)
    inp['filters'] = filters
    
    # Fluxes and uncertainties - already in units of maggies
    cols = [f'flux_{filt}' for filt in 'ugriz'] + [f'flux_w{n}' for n in range(1,5)]
    fluxes = row[cols].values.astype(float) / 3631

    # Errors
    cols_err = [f'{col}_e' for col in cols]
    errs = row[cols_err].values.astype(float) / 3631

    # Anything with a value of 9.999 is null, so mask those fluxes
    # TODO: check this
    inp['maggies'] = fluxes
    inp['maggies_unc'] = errs

    # This is an array of effective wavelengths for each of the filters.  
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    inp["phot_wave"] = np.array([f.wave_effective for f in inp["filters"]])
    inp["wavelength"] = None
    
    # Populate other fields with default
    inp = fix_obs(inp)
    return inp


# --------------
# Model Definition
# --------------

def build_model(var_redshift=False, fixed_metallicity=None, add_duste=False,
                add_neb=False, luminosity_distance=0.0, spec_z=0.0, **extras):
    
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param var_redshift:
        If True, allow redshift to vary. Otherwise, use known spec-z. 

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.

    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))

    : param spec_z: (optional)
        If var_redshift is False, known spectroscopic redshift from `build_obs`
        is input here
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]

    # Adjust model initial values (only important for optimization or emcee)
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.3
    model_params["tage"]["init"] = 13
    model_params["mass"]["init"] = 1e9

    # If we are going to be using emcee, it is useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be skipped
    model_params["mass"]["init_disp"] = 1e7
    model_params["tau"]["init_disp"] = 3.0
    model_params["tage"]["init_disp"] = 5.0
    model_params["tage"]["disp_floor"] = 2.0
    model_params["dust2"]["disp_floor"] = 0.1

    # adjust priors
    # model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    # model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
    # model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e10)

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity


    # If we know the redshift 
    if var_redshift:
        model_params["zred"]['isfree'] = True
        model_params["zred"]['init'] = 0.1
        model_params["zred"]['prior'] = priors.TopHat(mini=0,maxi=1)
    else:
        model_params["zred"]['isfree'] = False
        model_params["zred"]['init'] = spec_z 


    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    return model

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    obs = build_obs(**kwargs)
    model = build_model(spec_z=obs['redshift'], **kwargs)
    sps = build_sps(**kwargs)
    noise_model = build_noise(**kwargs)

    return obs, model, sps, noise_model


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    # parser.add_argument('--object_redshift', type=float, default=0.0,
    #                     help=("Redshift for the model"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--var_redshift', action="store_true",  
                        help="If set, make redshift a free parameter.")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    # Save the redshift to reconstruct later
    if not run_params['var_redshift']:
        run_params['spec_z'] = obs['redshift']

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = "output/{0}_{1}_result.h5".format(args.outfile, ts)

    output = fit_model(obs, model, sps, noise, **run_params)

    print("writing to {}".format(hfile))
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass
