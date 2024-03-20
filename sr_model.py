'''
SR_MODEL.PY: 
Model of the CRISPRa circuit with supplementary repression and the circuits it is compared with;
functions for analysing their performance.
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
import diffrax as diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController

# plotting and data handling
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, palettes as bkpalettes
from contourpy import contour_generator as cgen
import matplotlib as mpltlb

# miscellaneous
import time

# MODEL PARAMETERS -----------------------------------------------------------------------------------------------------
# default parameter values
def set_default_pars():
    par={} # initialise

    # most parameter values from Huang et al. 2021 (conversion formulae in brackets)

    # gRNA ODEs
    par['alpha_gi'] = 50*5*60  # maximum gRNA synthesis rate (Vmax[1/min] * pTg1 [nM] * 60 [min/h]=alpha_gi [nM/h])
    par['alpha_ga'] = par['alpha_gi']  # maximum gRNA synthesis rate (Vmax[1/min] * pTg1 [nM] * 60 [min/h]=alpha_ga [nM/h])
    par['alpha_gc'] = par['alpha_gi']  # maximum gRNA synthesis rate (Vmax[1/min] * pTg1 [nM] * 60 [min/h]=alpha_gc [nM/h])
    par['F_gi_0']=0.0 # basal extent of interfering gRNA synthesis (currently zero for simplicity)
    par['F_ga_0']=0.0 # basal extent of activating gRNA synthesis (currently zero for simplicity)
    par['F_gc_0']=0.0 # basal extent of competing gRNA synthesis (currently zero for simplicity)
    par['delta_gi'] = 0.2*60 # gRNA degradation rate [1/h] (theta [1/min] * 60 [min/h]=delta_gi [1/h])
    par['delta_ga'] = par['delta_gi']  # gRNA degradation rate [1/h] (theta [1/min] * 60 [min/h]=delta_ga [1/h])
    par['delta_gc'] = par['delta_gi']  # gRNA degradation rate [1/h] (theta [1/min] * 60 [min/h]=delta_gc [1/h])

    # CRISPR complex formation
    par['K_i'] = 0.01  # dissociation constant of interfering gRNA [nM] (identical parameter definition)
    par['K_a'] = par['K_i']  # dissociation constant of activating gRNA [nM] (identical parameter definition)
    par['K_c'] = par['K_i']  # dissociation constant of competing gRNA [nM] (identical parameter definition)
    par['Q_i'] = 0.5  # Hill coefficient of interfering gRNA (identical parameter definition)
    par['Q_a'] = par['Q_i']  # Hill coefficient of activating gRNA (identical parameter definition)
    par['Q_c'] = par['Q_i']  # Hill coefficient of competing gRNA (identical parameter definition)

    # target gene DNA concentrations
    par['D_p_tot'] = 200  # total target (output protein) gene DNA concentration [nM] (identical parameter definition)
    par['D_ct_tot'] = 200  # total target (competing gRNA's target) gene DNA concentration [nM] (identical parameter definition)

    # dCas9 ODE
    par['alpha_d'] = 0.6 * 30 * 60 # dCas9 synthesis rate [nM/h] (alpha_D [1/min] * p_D^t [nM] * 60 [min/h] = alpha_d [nM/h])

    # output protein ODE
    par['alpha_p'] = 1 * par['D_p_tot'] * 60  # output protein synthesis rate [nM/h] (alpha_4 [1/min] * DNA conc. [nM] * 60 [min/h] = alpha_p [nM/h])
    par['F_a_0'] = 1/20  # extent of leaky output protein synthesis [1/h] (FROM HO ET AL. 2020: 20-fold activation easily achievable)
    par['delta_p'] = 0 # assume no active degradation of output protein

    # cell growth rate
    par['lambda'] = 0.01*60  # gRNA degradation rate [1/h] (delta [1/min] * 60 [min/h]=lambda [1/h])

    # is CRISPR interference actually off-target? (1 if true, 0 if not)
    par['i_offtarget']=0.0
    # is CRISPR activation actually off-target? (1 if true, 0 if not)
    par['a_offtarget']=0.0

    # feedback controller - by default assumed not to be present
    par['fbck_present'] = 0  # 1 if feedback controller gRNA is present, 0 if not
    par['alpha_gf'] = 65 * 30 * 60  # maximum gRNA synthesis rate [nM/h] (u0 [1/min] * pT0 [nM] * 60 [min/h] = alpha_gf [nM/h])
    par['F_gf_0'] = 0.0  # basal extent of feedback controller gRNA synthesis (currently zero for simplicity)
    par['delta_gf'] = par['delta_gi']  # gRNA degradation rate [1/h] (theta [1/min] * 60 [min/h]=delta_gf [1/h])
    par['K_f'] = par['K_i']  # dissociation constant of feedback controller gRNA [nM] (identical parameter definition)
    par['Q_f'] = par['Q_i']  # Hill coefficient of feedback controller gRNA (identical parameter definition)
    par['D_d_tot'] = 30.0  # total target gene DNA concentration [nM] (identical parameter definition)

    return par

# ODE DEFINITION -------------------------------------------------------------------------------------------------------
def ode(t,x,args):
    # unpack arguments
    par=args[0] # model parameters

    # first three positions = extents of induction for interfering, activator and competing gRNAs
    ind=x[0:3]
    # next four positions = concentrations of interfering, activator and competing gRNAs; feedback controller gRNA
    g_i=x[3]
    g_a=x[4]
    g_c=x[5]
    g_f=x[6]
    # TOTAL dCas9 level
    d_tot = x[7]
    # output protein level
    p=x[8]

    # find total (free AND DNA-bound) amounts of CRISPR-gRNA complexes
    rc_denom=1+g_i/par['K_i']+g_a/par['K_a']+g_c/par['K_c']+g_f/par['K_f'] # resource competition denominator
    c_i_tot=d_tot * (g_i/par['K_i']) / rc_denom # interfering CRISPR complex
    c_a_tot=d_tot * (g_a/par['K_a']) / rc_denom # activating CRISPR complex
    c_c_tot=d_tot * (g_c/par['K_c']) / rc_denom # competing CRISPR complex
    c_f_tot=d_tot * (g_f/par['K_f']) / rc_denom # feedback controller CRISPR complex

    # find amounts of free (non-DNA-bound) CRISPR-gRNA complexes
    c_i=0.5*(
            (c_i_tot-par['D_p_tot']-par['Q_i']) +
            jnp.sqrt(jnp.square(par['D_p_tot']+par['Q_i']-c_i_tot) + 4*c_i_tot*par['Q_i'])
    )
    c_a=0.5*(
            (c_a_tot-par['D_p_tot']-par['Q_a']) +
            jnp.sqrt(jnp.square(par['D_p_tot']+par['Q_a']-c_a_tot) + 4*c_a_tot*par['Q_a'])
    )
    c_c=0.5*(
            (c_c_tot-par['D_ct_tot']-par['Q_c']) +
            jnp.sqrt(jnp.square(par['D_ct_tot']+par['Q_c']-c_c_tot) + 4*c_c_tot*par['Q_c'])
    )
    c_f=0.5*(
            (c_f_tot-par['D_d_tot']-par['Q_f']) +
            jnp.sqrt(jnp.square(par['D_d_tot']+par['Q_f']-c_f_tot) + 4*c_f_tot*par['Q_f'])
    )


    # return derivatives
    return jnp.array([0, 0, 0,  # induction levels constant
                      # interfering gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      reg_i(x, par, ind[0]) * par['alpha_gi'] - (par['delta_gi'] + par['lambda'])*g_i - par['lambda']*c_i_tot,
                      # activating gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      reg_a(x, par, ind[1]) * par['alpha_ga'] - (par['delta_ga'] + par['lambda'])*g_a - par['lambda']*c_a_tot,
                      # competing gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      reg_c(x, par, ind[2]) * par['alpha_gc'] - (par['delta_gc'] + par['lambda'])*g_c - par['lambda']*c_c_tot,
                      # feedback controller gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      par['fbck_present']*par['alpha_gf'] - (par['delta_gf'] + par['lambda'])*g_f - par['lambda']*c_f_tot,
                      # total dCas9 concentration: synthesis, dilution (dCas9 not actively degraded)
                      F_f(c_f,par) * par['alpha_d'] - par['lambda']*d_tot,
                      # output protein: synthesis, degradation and dilution
                      F_i(c_i, par) * F_a(c_a, par) * par['alpha_p'] - (par['delta_p'] + par['lambda'])*p
                      ])

# gRNA SYNTHESIS REGULATORY FUNCTIONS ----------------------------------------------------------------------------------
# regulation OF interfering gRNA
def reg_i(
        x,par,ind_i):    # state vector, system parameters, induction of inbitiory gRNA expression
    return par['F_gi_0']+(1-par['F_gi_0'])*ind_i


# regulation OF activating gRNA
def reg_a(
        x,par,ind_a):    # state vector, system parameters, induction of activating gRNA expression
    return par['F_ga_0']+(1-par['F_ga_0'])*ind_a


# regulation OF competing gRNA
def reg_c(
        x,par,ind_c):    # state vector, system parameters, induction of competing gRNA expression
    return par['F_gc_0']+(1-par['F_gc_0'])*ind_c


# PROTEIN SYNTHESIS REGULATORY FUNCTIONS ------------------------------------------------------------------------
# regulation by interfering CRISPR - just share of promoters not bound
def F_i(
        c_i, par):     # interfering CRISPR complex concentration, system parameters
    return 1*par['i_offtarget']+(1-par['i_offtarget'])*(
            1/(1+c_i/par['Q_i'])
    )


# regulation by activating CRISPR - share of promoters not bound + 'leakiness' in the form of not-bound-yet-active promoters
def F_a(
        c_a, par):  # activating CRISPR complex concentration, system parameters
    return 1 * par['a_offtarget'] + (1 - par['a_offtarget']) * (
            (par['F_a_0'] + c_a / par['Q_a']) / (1 + c_a / par['Q_a'])
    )

# regulation by feedback control CRISPR - share of promoters not bound + 'leakiness' in the form of not-bound-yet-active promoters
def F_f(
        c_f, par):     # activating CRISPR complex concentration, system parameters
    return 1/(1+c_f/par['Q_f'])

# FIND STEADY-STATE CRISPR REGULATION ----------------------------------------------------------------------------------
# all gRNAs - steady-state regulatory function values for a given set of inductions ind (and system parameters par)
@jax.jit
def induction_to_F(
        ind_mesh, par
):
    # set initial conditions
    x0_variables = jnp.array([0, 0, 0, 0,  # no gRNAs
                              1796,  # steady-state dCas9 level
                              0])  # no output protein
    x0s = jnp.concatenate((ind_mesh,x0_variables*jnp.ones((ind_mesh.shape[0],6))),axis=1)

    # simulation time axis parameters
    tf = (0, 48)  # simulation time frame

    # define arguments of the ODE term
    args = (
        par,  # system parameters
    )

    # ODE solver and its parameters
    term = ODETerm(ode)
    solver = diffrax.Kvaerno3()

    # define the time points at which we save the solution
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-6, dtmax=0.1)

    # solve the ODE
    get_sol = lambda x0: diffeqsolve(term, solver,
                                     args=args,
                                     t0=tf[0], t1=tf[1], dt0=0.1, y0=x0,
                                     max_steps=None,
                                     stepsize_controller=stepsize_controller)
    sol=jax.vmap(get_sol,in_axes=0)(x0s)

    # get output protein levels and regulatory function values
    p_ss, F_i_ss, F_a_ss, F_f_ss = jax.vmap(reconstruct_reg, in_axes=(0, 0, None))(sol.ts[:,-1],sol.ys[:,-1,:],par)
    return p_ss, F_i_ss, F_a_ss, jnp.multiply(F_i_ss,F_a_ss)


# system state vector to regulatory function values
@jax.jit
def reconstruct_reg(t,x,par):
    # unpack state vector
    ind = x[0:3]
    g_i = x[3]
    g_a = x[4]
    g_c = x[5]
    g_f=x[6]
    d_tot = x[7]
    p = x[8]

    # find total (free AND DNA-bound) amounts of CRISPR-gRNA complexes
    rc_denom=1+g_i/par['K_i']+g_a/par['K_a']+g_c/par['K_c']+g_f/par['K_f'] # resource competition denominator
    c_i_tot=d_tot * (g_i/par['K_i']) / rc_denom # interfering CRISPR complex
    c_a_tot=d_tot * (g_a/par['K_a']) / rc_denom # activating CRISPR complex
    c_c_tot=d_tot * (g_c/par['K_c']) / rc_denom
    c_f_tot=d_tot * (g_f/par['K_f']) / rc_denom

    # find amounts of free (non-DNA-bound) CRISPR-gRNA complexes
    c_i=0.5*(
            (c_i_tot-par['D_p_tot']-par['Q_i']) +
            jnp.sqrt(jnp.square(par['D_p_tot']+par['Q_i']-c_i_tot) + 4*c_i_tot*par['Q_i'])
    )
    c_a=0.5*(
            (c_a_tot-par['D_p_tot']-par['Q_a']) +
            jnp.sqrt(jnp.square(par['D_p_tot']+par['Q_a']-c_a_tot) + 4*c_a_tot*par['Q_a'])
    )
    c_c=0.5*(
            (c_c_tot-par['D_ct_tot']-par['Q_c']) +
            jnp.sqrt(jnp.square(par['D_ct_tot']+par['Q_c']-c_c_tot) + 4*c_c_tot*par['Q_c'])
    )
    c_f=0.5*(
            (c_f_tot-par['D_d_tot']-par['Q_f']) +
            jnp.sqrt(jnp.square(par['D_d_tot']+par['Q_f']-c_f_tot) + 4*c_f_tot*par['Q_f'])
    )

    # find regulatory function values
    F_i_reconstructed=F_i(c_i,par)
    F_a_reconstructed=F_a(c_a,par)
    F_f_reconstructed=F_f(c_f,par)

    # return output protein level and regulatory function values
    return p, F_i_reconstructed, F_a_reconstructed, F_f_reconstructed


# PERFORMANCE METRICS --------------------------------------------------------------------------------------------------
# robustness of ACTIVATION: integral of sum of squared differences over the log-scale induction axis
def rob(val_ss_with, val_ss_no, inda_axis):
    # if a single doise-response curve given, output is a single value
    if (len(val_ss_with.shape) == 1):
        log_inda_axis = np.log(inda_axis)  # get the log-axis
        sos_diff = np.square(val_ss_with - val_ss_no)  # get the sum of squared differences
        return 1 / np.sum((sos_diff[0:-1]+sos_diff[1:])/2 * (log_inda_axis[1:] - log_inda_axis[0:-1]))
    # for multiple dose-response curves, output is a vector of values
    else:
        log_inda_axis = np.ones((val_ss_with.shape[0],1))*np.log(inda_axis)  # get the log-axis
        sos_diff = np.square(val_ss_with - val_ss_no)
        return 1 / np.sum((sos_diff[:,0:-1]+sos_diff[:,1:])/2 * (log_inda_axis[:,1:] - log_inda_axis[:,0:-1]), axis=1)


# change in half-saturation constants with and without competition
def change_in_K(val_ss_with, val_ss_no, inda_axis):
    # if a single doise-response curve given, output is a single value
    if (len(val_ss_with.shape) == 1):
        # find the halfway points in the activation curves
        halfval_no = 0.5#(np.max(val_ss_no) + np.min(val_ss_no)) / 2
        halfval_with = 0.5#(np.max(val_ss_with) + np.min(val_ss_with)) / 2

        # find the indices of half-saturation constants in inda_axis
        K_no_arg = int(np.argwhere(val_ss_no >= halfval_no)[0])
        K_with_arg = int(np.argwhere(val_ss_with >= halfval_with)[0])

        # find the fold change in half-saturation constants
        K_fold_change = inda_axis.take(K_with_arg) / inda_axis.take(K_no_arg)

        # return the change in half-saturation constants and the arguments of constants in inda_axis
        return K_fold_change, K_no_arg, K_with_arg
    # for multiple dose-response curves, output is a vector of values
    else:
        K_no_arg = np.zeros(val_ss_no.shape[0], dtype=int)
        K_with_arg = np.zeros(val_ss_no.shape[0], dtype=int)
        K_fold_change = np.zeros(val_ss_no.shape[0])
        for i in range(0, val_ss_no.shape[0]):
            halfval_no = 0.5#(np.max(val_ss_no[i, :]) + np.min(val_ss_no[i, :])) / 2
            halfval_with = 0.5#(np.max(val_ss_with[i, :]) + np.min(val_ss_with[i, :])) / 2

            K_no_arg[i] = np.argmax(val_ss_no[i, :] >= halfval_no)
            K_with_arg[i] = np.argmax(val_ss_with[i, :] >= halfval_with)

            K_fold_change[i] = inda_axis[K_with_arg[i]] / inda_axis[K_no_arg[i]]

        # return the change in half-saturation constants and the arguments of constants in inda_axis
        return K_fold_change, K_no_arg, K_with_arg