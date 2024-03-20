'''
CISTRANS_MODEL.PY:
Model of the cis and trans feedback controllers and the open loop system they are compared with;
functions for analysing their performance.
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import jaxopt
import functools
import diffrax as diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, SteadyStateEvent

# plotting and data handling
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, palettes as bkpalettes
from contourpy import contour_generator as cgen
import matplotlib as mpltlb

# miscellaneous
import time

# import custom functions for AsiA modelling
from sr_model import reg_i, reg_a, reg_c
from sr_model import F_i, F_a, F_f

# MODEL PARAMETERS -----------------------------------------------------------------------------------------------------
# default parameter values
def set_default_pars_cistrans():
    par = {}  # initialise

    # most parameter values from Huang et al. 2021 (conversion formulae in brackets)

    # gRNA ODEs
    par['alpha_gi'] = 50 * 200 * 60  # maximum gRNA synthesis rate (Vmax[1/min] * pTg1 [nM] * 60 [min/h]=alpha_gi [nM/h])
    par['alpha_ga'] = par['alpha_gi']  # maximum gRNA synthesis rate (Vmax[1/min] * pTg1 [nM] * 60 [min/h]=alpha_ga [nM/h])
    par['alpha_gc'] = par['alpha_gi']  # maximum gRNA synthesis rate (Vmax[1/min] * pTg1 [nM] * 60 [min/h]=alpha_gc [nM/h])
    par['F_gi_0'] = 0.0  # basal extent of interfering gRNA synthesis (currently zero for simplicity)
    par['F_ga_0'] = 0.0  # basal extent of activating gRNA synthesis (currently zero for simplicity)
    par['F_gc_0'] = 0.0  # basal extent of competing gRNA synthesis (currently zero for simplicity)
    par['delta_gi'] = 0.2 * 60  # gRNA degradation rate [1/h] (theta [1/min] * 60 [min/h]=delta_gi [1/h])
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
    par['D_i_tot'] = 200 # total target (interfering gRNA AND output protein) gene DNA concentration [nM] (identical parameter definition)
    par['D_a_tot'] = 200  # total target (activating gRNA) gene DNA concentration [nM] (identical parameter definition)
    par['D_ct_tot'] = 200  # total target (competing gRNA's target) gene DNA concentration [nM] (identical parameter definition))

    # dCas9 ODE
    par['alpha_d'] = 0.6 * 30 * 60  # dCas9 synthesis rate [nM/h] (alpha_D [1/min] * p_D^t [nM] * 60 [min/h] = alpha_d [nM/h])

    # output protein ODE - same gene as interfering gRNA => D_p_tot=D_i_tot!
    par['alpha_p'] = 1 * par['D_i_tot'] * 60  # output protein synthesis rate [nM/h] (alpha_4 [1/min] * DNA conc. [nM] * 60 [min/h] = alpha_p [nM/h])
    par['F_a_0'] = 1 / 20  # extent of leaky output protein synthesis [1/h] (FROM HO ET AL. 2020: 20-fold activation easily achievable)
    par['delta_p'] = 0  # assume no active degradation of output protein

    # cell growth rate
    par['lambda'] = 0.01 * 60  # gRNA degradation rate [1/h] (delta [1/min] * 60 [min/h]=lambda [1/h])

    # is CRISPR interference actually off-target? (1 if true, 0 if not)
    par['i_offtarget'] = 0.0
    # is CRISPR activation actually off-target? (1 if true, 0 if not)
    par['a_offtarget'] = 0.0

    return par

# ODE DEFINITION -------------------------------------------------------------------------------------------------------
# CIS FEEDBACK: interfering sgRNA represses itself
def ode_cis(t,x,args):
    # unpack arguments
    par=args[0] # model parameters

    # first three positions = extents of induction for interfering, activator and competing gRNAs
    ind=x[0:3]
    # next four positions = concentrations of interfering, activator and competing gRNAs; feedback controller gRNA
    g_i=x[3]
    g_a=x[4]    # zero but kept in the ODEs for consistency
    g_c=x[5]
    g_f=x[6]    # zero but kept in the ODEs for consistency
    # TOTAL dCas9 level
    d_tot = x[7]
    # output protein level
    p=x[8]

    # find total (free AND DNA-bound) amounts of CRISPR-gRNA complexes
    rc_denom=1+g_i/par['K_i']+0+g_c/par['K_c']+0 # resource competition denominator
    c_i_tot=d_tot * (g_i/par['K_i']) / rc_denom # interfering CRISPR complex
    c_a_tot=0   #d_tot * (g_a/par['K_a']) / rc_denom # activating CRISPR complex
    c_c_tot=d_tot * (g_c/par['K_c']) / rc_denom # competing CRISPR complex
    c_f_tot=0   #d_tot * (g_f/par['K_f']) / rc_denom # feedback controller CRISPR complex

    # find amounts of free (non-DNA-bound) CRISPR-gRNA complexes
    c_i=0.5*(
            (c_i_tot-par['D_i_tot']-par['Q_i']) +
            jnp.sqrt(jnp.square(par['D_i_tot']+par['Q_i']-c_i_tot) + 4*c_i_tot*par['Q_i'])
    )
    c_a=0
    # c_a=0.5*(
    #         (c_a_tot-par['D_ia_tot']-par['Q_a']) +
    #         jnp.sqrt(jnp.square(par['D_ia_tot']+par['Q_a']-c_a_tot) + 4*c_a_tot*par['Q_a'])
    # )
    c_c=0.5*(
            (c_c_tot-par['D_ct_tot']-par['Q_c']) +
            jnp.sqrt(jnp.square(par['D_ct_tot']+par['Q_c']-c_c_tot) + 4*c_c_tot*par['Q_c'])
    )
    c_f=0
    # c_f=0.5*(
    #         (c_f_tot-par['D_ia_tot']-par['Q_f']) +
    #         jnp.sqrt(jnp.square(par['D_ia_tot']+par['Q_f']-c_f_tot) + 4*c_f_tot*par['Q_f'])
    # )


    # return derivatives
    return jnp.array([0, 0, 0,  # induction levels constant
                      # interfering gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      F_i(c_i, par) * reg_i(x, par, ind[0]) * par['alpha_gi'] - (par['delta_gi'] + par['lambda'])*g_i - par['lambda']*c_i_tot,
                      # activating gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      0,    #reg_a(x, par, ind[1]) * par['alpha_ga'] - (par['delta_ga'] + par['lambda'])*g_a - par['lambda']*c_a_tot,
                      # competing gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      reg_c(x, par, ind[2]) * par['alpha_gc'] - (par['delta_gc'] + par['lambda'])*g_c - par['lambda']*c_c_tot,
                      # feedback controller gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      0,    #par['fbck_present']*par['alpha_gf'] - (par['delta_gf'] + par['lambda'])*g_f - par['lambda']*c_f_tot,
                      # total dCas9 concentration: synthesis, dilution (dCas9 not actively degraded)
                      par['alpha_d'] - par['lambda']*d_tot,
                      # output protein: synthesis, degradation and dilution
                      F_i(c_i, par) * reg_i(x, par, ind[0]) * par['alpha_p'] - (par['delta_p'] + par['lambda'])*p
                      ])

# CIS FEEDBACK: interfering sgRNA and activating sgRNA regulate each other
def ode_trans(t,x,args):
    # unpack arguments
    par = args[0]  # model parameters

    # first three positions = extents of induction for interfering, activator and competing gRNAs
    ind = x[0:3]
    # next four positions = concentrations of interfering, activator and competing gRNAs; feedback controller gRNA
    g_i = x[3]
    g_a = x[4]
    g_c = x[5]
    g_f = x[6]
    # TOTAL dCas9 level
    d_tot = x[7]
    # output protein level
    p = x[8]

    # find total (free AND DNA-bound) amounts of CRISPR-gRNA complexes
    rc_denom = 1 + g_i / par['K_i'] + g_a / par['K_a'] + g_c / par['K_c'] + 0  # resource competition denominator
    c_i_tot = d_tot * (g_i / par['K_i']) / rc_denom  # interfering CRISPR complex
    c_a_tot = d_tot * (g_a / par['K_a']) / rc_denom  # activating CRISPR complex
    c_c_tot = d_tot * (g_c / par['K_c']) / rc_denom  # competing CRISPR complex
    c_f_tot = 0     #d_tot * (g_f / par['K_f']) / rc_denom  # feedback controller CRISPR complex

    # find amounts of free (non-DNA-bound) CRISPR-gRNA complexes
    c_i = 0.5 * (
            (c_i_tot - par['D_a_tot'] - par['Q_i']) +
            jnp.sqrt(jnp.square(par['D_a_tot'] + par['Q_i'] - c_i_tot) + 4 * c_i_tot * par['Q_i'])
    )
    c_a = 0.5 * (
            (c_a_tot - par['D_i_tot'] - par['Q_a']) +
            jnp.sqrt(jnp.square(par['D_i_tot'] + par['Q_a'] - c_a_tot) + 4 * c_a_tot * par['Q_a'])
    )
    c_c = 0.5 * (
            (c_c_tot - par['D_ct_tot'] - par['Q_c']) +
            jnp.sqrt(jnp.square(par['D_ct_tot'] + par['Q_c'] - c_c_tot) + 4 * c_c_tot * par['Q_c'])
    )
    c_f = 0
    # c_f = 0.5 * (
    #         (c_f_tot - par['D_d_tot'] - par['Q_f']) +
    #         jnp.sqrt(jnp.square(par['D_d_tot'] + par['Q_f'] - c_f_tot) + 4 * c_f_tot * par['Q_f'])
    # )

    # return derivatives
    return jnp.array([0, 0, 0,  # induction levels constant
                      # interfering gRNA (regulated by activating gRNA): synthesis, degradation and dilution, net outflow due to dCas9 binding
                      F_a(c_a, par) * reg_i(x, par, ind[0]) * par['alpha_gi'] - (par['delta_gi'] + par['lambda']) * g_i - par['lambda'] * c_i_tot,
                      # activating gRNA (regulated by interfering gRNA): synthesis, degradation and dilution, net outflow due to dCas9 binding
                      F_i(c_i, par) * reg_a(x, par, ind[1]) * par['alpha_ga'] - (par['delta_ga'] + par['lambda']) * g_a - par['lambda'] * c_a_tot,
                      # competing gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      reg_c(x, par, ind[2]) * par['alpha_gc'] - (par['delta_gc'] + par['lambda']) * g_c - par['lambda'] * c_c_tot,
                      # feedback controller gRNA: synthesis, degradation and dilution, net outflow due to dCas9 binding
                      0,    #par['fbck_present'] * par['alpha_gf'] - (par['delta_gf'] + par['lambda']) * g_f - par['lambda'] * c_f_tot,
                      # total dCas9 concentration: synthesis, dilution (dCas9 not actively degraded)
                      par['alpha_d'] - par['lambda'] * d_tot,
                      # output protein: synthesis, degradation and dilution
                      F_a(c_a, par) * reg_i(x,par, ind[0]) * par['alpha_p'] - (par['delta_p'] + par['lambda']) * p
                      ])

# FIND STEADY-STATE CRISPR REGULATION ----------------------------------------------------------------------------------
# CIS
# all gRNAs - steady-state regulatory function values for a given set of inductions ind (and system parameters par)
@jax.jit
def induction_to_p_and_F_cis(
        ind_mesh, par):
    # set initial conditions
    x0_variables = jnp.array([0, 0, 0, 0,  # no gRNAs
                              1796,  # steady-state dCas9 level
                              0])  # no output protein
    x0s = jnp.concatenate((ind_mesh,x0_variables*jnp.ones((ind_mesh.shape[0],6))),axis=1)

    # simulation time axis parameters
    tf = (0, 24)  # simulation time frame

    # define the ODE term
    term = ODETerm(ode_cis)
    # define arguments of the ODE term
    args = (
        par,  # system parameters
    )

    # ODE solver and its parameters
    solver = diffrax.Kvaerno3()

    # define the time points at which we save the solution
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-6, dtmax=0.1)

    # solve the ODE
    get_sol = lambda x0: diffeqsolve(term, solver,
                                     args=args,
                                     t0=tf[0], t1=tf[1], dt0=0.1, y0=x0,
                                     max_steps=None,stepsize_controller=stepsize_controller)
    sol=jax.vmap(get_sol,in_axes=0)(x0s)

    # get regulatory function values
    p, F_i_ss = jax.vmap(reconstruct_p_and_reg_cis, in_axes=(0, 0, None))(sol.ts[:,-1],sol.ys[:,-1,:],par)
    return p, F_i_ss


# system state vector to regulatory function values
@jax.jit
def reconstruct_p_and_reg_cis(t,x,par):
    # unpack state vector
    ind = x[0:3]
    g_i = x[3]
    g_a = x[4]
    g_c = x[5]
    g_f=x[6]
    d_tot = x[7]
    p = x[8]

    # find total (free AND DNA-bound) amounts of CRISPR-gRNA complexes
    rc_denom=1+g_i/par['K_i']+0+g_c/par['K_c']+0 # resource competition denominator
    c_i_tot=d_tot * (g_i/par['K_i']) / rc_denom # interfering CRISPR complex
    c_a_tot=0   #d_tot * (g_a/par['K_a']) / rc_denom # activating CRISPR complex
    c_c_tot=d_tot * (g_c/par['K_c']) / rc_denom
    c_f_tot=0   #d_tot * (g_f/par['K_f']) / rc_denom

    # find amounts of free (non-DNA-bound) CRISPR-gRNA complexes
    c_i=0.5*(
            (c_i_tot-par['D_i_tot']-par['Q_i']) +
            jnp.sqrt(jnp.square(par['D_i_tot']+par['Q_i']-c_i_tot) + 4*c_i_tot*par['Q_i'])
    )
    c_a=0
    # c_a=0.5*(
    #         (c_a_tot-par['D_p_tot']-par['Q_a']) +
    #         jnp.sqrt(jnp.square(par['D_p_tot']+par['Q_a']-c_a_tot) + 4*c_a_tot*par['Q_a'])
    # )
    c_c=0.5*(
            (c_c_tot-par['D_ct_tot']-par['Q_c']) +
            jnp.sqrt(jnp.square(par['D_ct_tot']+par['Q_c']-c_c_tot) + 4*c_c_tot*par['Q_c'])
    )
    c_f=0
    # c_f=0.5*(
    #         (c_f_tot-par['D_d_tot']-par['Q_f']) +
    #         jnp.sqrt(jnp.square(par['D_d_tot']+par['Q_f']-c_f_tot) + 4*c_f_tot*par['Q_f'])
    # )

    # find regulatory function values
    F_i_reconstructed=F_i(c_i,par)

    return p, F_i_reconstructed


# TRANS
# all gRNAs - steady-state regulatory function values for a given set of inductions ind (and system parameters par)
@jax.jit
def induction_to_p_and_F_trans(
        ind_mesh, par):
    # set initial conditions
    x0_variables = jnp.array([0, 0, 0, 0,  # no gRNAs
                              1796,  # steady-state dCas9 level
                              10])  # no output protein
    x0s = jnp.concatenate((ind_mesh, x0_variables * jnp.ones((ind_mesh.shape[0], 6))), axis=1)

    # simulation time axis parameters
    tf = (0, 24)  # simulation time frame

    # define the ODE term
    term = ODETerm(ode_trans)
    # define arguments of the ODE term
    args = (
        par,  # system parameters
    )

    # ODE solver and its parameters
    solver = diffrax.Euler()

    # define the time points at which we save the solution
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-6, dtmax=0.1)

    # solve the ODE
    get_sol = lambda x0: diffeqsolve(term, solver,
                                     args=args,
                                     t0=tf[0], t1=tf[1], dt0=1e-5, y0=x0,
                                     max_steps=None)    #,
                                     # stepsize_controller=stepsize_controller)
    sol = jax.vmap(get_sol, in_axes=0)(x0s)

    # get regulatory function values
    p_ss, F_i_ss, F_a_ss = jax.vmap(reconstruct_p_and_reg_trans, in_axes=(0, 0, None))(sol.ts[:, -1], sol.ys[:, -1, :], par)
    return p_ss, F_i_ss, F_a_ss


# system state vector to regulatory function values
@jax.jit
def reconstruct_p_and_reg_trans(t, x, par):
    # unpack state vector
    ind = x[0:3]
    g_i = x[3]
    g_a = x[4]
    g_c = x[5]
    g_f = x[6]
    d_tot = x[7]
    p = x[8]

    # find total (free AND DNA-bound) amounts of CRISPR-gRNA complexes
    rc_denom = 1 + g_i / par['K_i'] + g_a / par['K_a'] + g_c / par['K_c'] + 0  # resource competition denominator
    c_i_tot = d_tot * (g_i / par['K_i']) / rc_denom  # interfering CRISPR complex
    c_a_tot = d_tot * (g_a / par['K_a']) / rc_denom  # activating CRISPR complex
    c_c_tot = d_tot * (g_c / par['K_c']) / rc_denom
    c_f_tot = 0     #d_tot * (g_f / par['K_f']) / rc_denom

    # find amounts of free (non-DNA-bound) CRISPR-gRNA complexes
    c_i = 0.5 * (
            (c_i_tot - par['D_a_tot'] - par['Q_i']) +
            jnp.sqrt(jnp.square(par['D_a_tot'] + par['Q_i'] - c_i_tot) + 4 * c_i_tot * par['Q_i'])
    )
    c_a = 0.5 * (
            (c_a_tot - par['D_i_tot'] - par['Q_a']) +
            jnp.sqrt(jnp.square(par['D_i_tot'] + par['Q_a'] - c_a_tot) + 4 * c_a_tot * par['Q_a'])
    )
    c_c = 0.5 * (
            (c_c_tot - par['D_ct_tot'] - par['Q_c']) +
            jnp.sqrt(jnp.square(par['D_ct_tot'] + par['Q_c'] - c_c_tot) + 4 * c_c_tot * par['Q_c'])
    )#
    c_f=0
    # c_f = 0.5 * (
    #         (c_f_tot - par['D_d_tot'] - par['Q_f']) +
    #         jnp.sqrt(jnp.square(par['D_d_tot'] + par['Q_f'] - c_f_tot) + 4 * c_f_tot * par['Q_f'])
    # )

    # find regulatory function values
    F_i_reconstructed = F_i(c_i, par)
    F_a_reconstructed = F_a(c_a, par)

    return p, F_i_reconstructed, F_a_reconstructed


# FIND DYNAMIC PERFORMANCE METRICS -------------------------------------------------------------------------------------
# resource perturbations
@functools.partial(jax.jit, static_argnums=(2,3))
def induction_to_dynmetrics_res(ind_mesh,par,term,pert_char,settle_bound=0.05):
    # unpack the perturbance characterisation
    tf_pre=pert_char[0]
    tf_pulse=pert_char[1]
    tf_post=pert_char[2]
    ind_c_pert=pert_char[3]

    # get default competitor induction from initial condition
    ind_c_default=ind_mesh[0,2]

    # set initial conditions
    x0_variables = jnp.array([0, 0, 0, 0,  # no gRNAs
                              1796,  # steady-state dCas9 level
                              10])  # no output protein
    x0s = jnp.concatenate((ind_mesh, x0_variables * jnp.ones((ind_mesh.shape[0], 6))), axis=1)

    # ODE solver parameters
    solver = diffrax.Kvaerno3()
    savetimestep = 0.01
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-6, dtmax=0.01)
    get_sol = lambda x0, tf: diffeqsolve(term, solver,
                      args=(par,),
                      t0=tf[0], t1=tf[1], dt0=0.1, y0=x0,
                      saveat=SaveAt(ts=jnp.arange(tf[0], tf[1], savetimestep)),
                      max_steps=None,
                      stepsize_controller=stepsize_controller)

    # solve the ODE BEFORE the competitor is induced
    sol_pre = jax.vmap(get_sol, in_axes=(0, None))(x0s,tf_pre)
    xs_pre=sol_pre.ys
    ts_pre=sol_pre.ts
    ps_pre = xs_pre[:, :, 8]
    x_ss_unperts = sol_pre.ys[:,-1, :]
    p_ss_unperts = x_ss_unperts[:,8]

    # solve the ODE DURING the competitor is induced
    x0s_pulse = sol_pre.ys[:,-1, :].at[:,2].set(ind_c_pert)
    sol_pulse = jax.vmap(get_sol, in_axes=(0, None))(x0s_pulse,tf_pulse)
    xs_pulse=sol_pulse.ys
    ts_pulse=sol_pulse.ts
    ps_pulse = xs_pulse[:, :, 8]

    # solve the ODE AFTER the competitor is induced
    x0s_post = sol_pulse.ys[:,-1, :].at[:,2].set(ind_c_default)
    sol_post = jax.vmap(get_sol, in_axes=(0, None))(x0s_post,tf_post)
    xs_post=sol_post.ys
    ts_post=sol_post.ts
    ps_post = xs_post[:, :, 8]

    # concatenate system trajectories
    xs=jnp.concatenate((xs_pre,xs_pulse,xs_post),axis=1)
    ts=jnp.concatenate((ts_pre,ts_pulse,ts_post),axis=1)
    ps=jnp.concatenate((ps_pre,ps_pulse,ps_post),axis=1)

    #  FIND ROBUSTNESS AND SPEED OF RECOVERY
    ts_pulseandpost=jnp.concatenate((ts_pulse,ts_post),axis=1)
    ps_pulseandpost=jnp.concatenate((ps_pulse,ps_post),axis=1)
    p_ss_unperts_stacked=jnp.tile(p_ss_unperts,(ps_pulseandpost.shape[1],1)).T

    # peak absolute disturbance
    peakdists=jnp.max(jnp.abs(ps_pulseandpost - p_ss_unperts_stacked), axis=1)

    # peak relative disturbance/fold-change
    ps_fold_changes= jnp.divide(ps_pulseandpost,p_ss_unperts_stacked)
    peakdists_fold_increase=jnp.max(ps_fold_changes,axis=1)
    peakdists_fold_decrease=1/jnp.min(ps_fold_changes,axis=1)
    peakdists_fold=jnp.where(jnp.abs(peakdists_fold_increase) > jnp.abs(peakdists_fold_decrease),peakdists_fold_increase,peakdists_fold_decrease)

    # time to recover to be within 5% of steady state
    within_settle_bound=jnp.logical_and(ts_pulseandpost > tf_post[0],jnp.abs(ps_pulseandpost - p_ss_unperts_stacked) < p_ss_unperts_stacked * settle_bound)
    all_after_within_settle_bound=jnp.flip(jnp.cumprod(jnp.flip(within_settle_bound,axis=1),axis=1),axis=1)
    previous_within_settle_bound=jnp.roll(within_settle_bound,1,axis=1).at[:,0].set(True)
    settling_point=jnp.logical_and(all_after_within_settle_bound,jnp.logical_not(previous_within_settle_bound))
    rectime_indices=ts_pre.shape[1]+jnp.argmax(settling_point,axis=1)

    return peakdists, peakdists_fold, rectime_indices, xs, ts


# transcriptional perturbations
@functools.partial(jax.jit, static_argnums=(2,3))
def induction_to_dynmetrics_transc(ind_mesh,par,term,pert_char,settle_bound=0.05):
    # unpack the perturbance characterisation
    tf_pre=pert_char[0]
    tf_pulse=pert_char[1]
    tf_post=pert_char[2]
    ind_a_pert_relative=pert_char[3]

    # set initial conditions
    x0_variables = jnp.array([0, 0, 0, 0,  # no gRNAs
                              1796,  # steady-state dCas9 level
                              10])  # no output protein
    x0s = jnp.concatenate((ind_mesh, x0_variables * jnp.ones((ind_mesh.shape[0], 6))), axis=1)

    # ODE solver parameters
    solver = diffrax.Kvaerno3()
    savetimestep = 0.01
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-6, dtmax=0.01)
    get_sol = lambda x0, tf: diffeqsolve(term, solver,
                      args=(par,),
                      t0=tf[0], t1=tf[1], dt0=0.1, y0=x0,
                      saveat=SaveAt(ts=jnp.arange(tf[0], tf[1], savetimestep)),
                      max_steps=None,
                      stepsize_controller=stepsize_controller)

    # solve the ODE BEFORE the competitor is induced
    sol_pre = jax.vmap(get_sol, in_axes=(0, None))(x0s,tf_pre)
    xs_pre=sol_pre.ys
    ts_pre=sol_pre.ts
    ps_pre = xs_pre[:, :, 8]
    x_ss_unperts = sol_pre.ys[:,-1, :]
    p_ss_unperts = x_ss_unperts[:,8]

    # solve the ODE DURING the competitor is induced
    x0s_pulse = sol_pre.ys[:,-1, :].at[:,0].set(x_ss_unperts[:,0]*ind_a_pert_relative)
    sol_pulse = jax.vmap(get_sol, in_axes=(0, None))(x0s_pulse,tf_pulse)
    xs_pulse=sol_pulse.ys
    ts_pulse=sol_pulse.ts
    ps_pulse = xs_pulse[:, :, 8]

    # solve the ODE AFTER the competitor is induced
    x0s_post = sol_pulse.ys[:,-1, :].at[:,0].set(x_ss_unperts[:,0])
    sol_post = jax.vmap(get_sol, in_axes=(0, None))(x0s_post,tf_post)
    xs_post=sol_post.ys
    ts_post=sol_post.ts
    ps_post = xs_post[:, :, 8]

    # concatenate system trajectories
    xs=jnp.concatenate((xs_pre,xs_pulse,xs_post),axis=1)
    ts=jnp.concatenate((ts_pre,ts_pulse,ts_post),axis=1)
    ps=jnp.concatenate((ps_pre,ps_pulse,ps_post),axis=1)

    #  FIND ROBUSTNESS AND SPEED OF RECOVERY
    ts_pulseandpost=jnp.concatenate((ts_pulse,ts_post),axis=1)
    ps_pulseandpost=jnp.concatenate((ps_pulse,ps_post),axis=1)
    p_ss_unperts_stacked=jnp.tile(p_ss_unperts,(ps_pulseandpost.shape[1],1)).T

    # peak absolute disturbance
    peakdists = jnp.max(jnp.abs(ps_pulseandpost - p_ss_unperts_stacked),axis=1)

    # peak relative disturbance/fold-change
    ps_fold_changes= jnp.divide(ps_pulseandpost,p_ss_unperts_stacked)
    peakdists_fold_increase = jnp.max(ps_fold_changes, axis=1)
    peakdists_fold_decrease = 1/jnp.min(ps_fold_changes, axis=1)
    peakdists_fold = jnp.where(jnp.abs(peakdists_fold_increase) > jnp.abs(peakdists_fold_decrease),
                               peakdists_fold_increase, peakdists_fold_decrease)

    # time to recover to be within 5% of steady state
    within_settle_bound=jnp.logical_and(ts_pulseandpost > tf_post[0],jnp.abs(ps_pulseandpost - p_ss_unperts_stacked) < p_ss_unperts_stacked * settle_bound)
    all_after_within_settle_bound=jnp.flip(jnp.cumprod(jnp.flip(within_settle_bound,axis=1),axis=1),axis=1)
    previous_within_settle_bound=jnp.roll(within_settle_bound,1,axis=1).at[:,0].set(True)
    settling_point=jnp.logical_and(all_after_within_settle_bound,jnp.logical_not(previous_within_settle_bound))
    rectime_indices=ts_pre.shape[1]+jnp.argmax(settling_point,axis=1)

    return peakdists, peakdists_fold, rectime_indices, xs, ts
