# -*- coding: utf-8 -*-
"""
author: Victor Fernandez

Collection of functions building within the many-body formalismthe hamiltonian
of the Kitaev chain embedded into a single-mode photonic cavity.
"""
# This file is part of the repository "KitaevChain_CavityPhotons"
# The file is subject to the license terms in the file LICENSE found
# in the top-level directory of the repository.

# Updates:
# Repeated computations in the hopping and superconducting matrices
# saturate the code. To improve performance two new functions have been added:
#    - hopping_indexes()
#    - super_indexes()
# and one function has been upgraded:
#    - g_coup_terms()

import numpy as np
import math 
import itertools
from math import factorial
from math import comb
from scipy.sparse import coo_array, bmat, eye
from scipy.sparse.linalg import eigsh
from scipy.sparse import identity
from scipy.linalg import eigh
import scipy.sparse.linalg as sla
import h5py
import time

import matplotlib.pyplot as plt

def v_ocup_indx(n_sites,occup):
    """ function to calculate the vector basis with a given occupation and
    for spinless fermions in a chain with n sites """
    # INPUT:
    # n_sites (integer): number of sites in the chain
    # occup (integer):  number of electrons (occupancy) in  the subspace
    # OUTPUT:
    # v (2D array): the rows of the array, with dimension n_sites, are
    #               the basis vectors of the supspace with a number of
    #               particles given by occup

    block_dim = math.comb(n_sites,occup)
    
    v = np.zeros((block_dim,n_sites), dtype=np.int8)
    sites_array = [ii for ii in range(n_sites)]

    nn = 0
    for indxs in itertools.combinations(sites_array,occup):
        for ll in indxs:
            v[nn,ll] = 1
            
        nn += 1
            
    return v

def sq_combi(n,m,k):
    """ function calculating the products of factorials appearing in polynomials J_{mn} """
    # INPUT
    # n (integer):  photon occupation of subspace
    # m (integer):  photon occupation of subspace
    # k (integer):  (0 < k <= n) or (0 < k <= m)
    # OUTPUT
    # result (float):

    if n >= m:
        result = np.sqrt(factorial(n))*np.sqrt(factorial(m))/factorial(k)/factorial(k+n-m)
        return result
    else:
        result = np.sqrt(factorial(n))*np.sqrt(factorial(m))/factorial(k)/factorial(k+m-n)
        return result

def poly_J(n,m,x):
    """ function calculating the polynomial J_{mn} """
    # INPUT
    # n (integer):  photon occupation of subspace
    # m (integer):  photon occupation of subspace
    # x (float):    argument at which J_{nm} is evaluated
    # OUTPUT
    # result (float):
    
    if n >= m:
        terms = [(-1)**k*sq_combi(n,m,k)*x**(2*k+n-m)/factorial(m-k) for k in range(0,m+1)]

    else:
        terms = [(-1)**k*sq_combi(n,m,k)*x**(2*k+m-n)/factorial(n-k) for k in range(0,n+1)]

    return np.sum(np.array(terms))

def hopping(t_hop,n_sites,filling,parity=None, j_neigh=1):
    """ function to calculate the hopping (homogeneous) in a lattice """
    # INPUT
    # t_hop (float):      the hopping
    # n_sites (integer):  number of sites in the chain
    # filling (integer):  filling factor
    # parity (string):    even, odd or None
    # j_neigh (integer):  degree of neighbor, nearest neighbor (NN) or next NN
    # OUTPUT
    # result (sparse array):
    
    row = []
    column = []
    value = []

    if parity == None:
        occu_list = [oo for oo in range(0,filling+1,1)]
    elif parity == 'even':
        occu_list = [oo for oo in range(0,filling+1,2)]
    elif parity == 'odd':
        occu_list = [oo for oo in range(1,filling+1,2)]
    else:
        print('Error: parity not recognized')
        
    o_sub_dim = 0
    for oo in occu_list:
        v_blocks = v_ocup_indx(n_sites,oo)
        for jj in range(n_sites-j_neigh):
    
            v_indx_neigh_r = np.where((v_blocks[::,jj] == 1) & (v_blocks[::,jj+j_neigh]== 0))
            v_indx_neigh_l = np.where((v_blocks[::,jj] == 0) & (v_blocks[::,jj+j_neigh]== 1))
            #print(v_indx_neigh_r)
            #print(v_indx_neigh_l)
    
            for mm, nn in itertools.product(v_indx_neigh_r[0],v_indx_neigh_l[0]):
    
                short_va = v_blocks[mm,::]
                short_vb = v_blocks[nn,::]
                short_va = np.delete(short_va,[jj,jj+j_neigh])
                short_vb = np.delete(short_vb,[jj,jj+j_neigh])
                
                diff = short_va - short_vb
                if np.dot(diff,diff) == 0:
                    row.append(nn+o_sub_dim)
                    column.append(mm+o_sub_dim)              
                    value.append(-t_hop*(-1)**(v_blocks[mm,jj+1]))

        o_sub_dim = o_sub_dim + len(v_blocks)

    return coo_array((value,(row, column)), shape=(o_sub_dim,o_sub_dim))

def hopping_indexes(n_sites,filling,parity):
    r""" function calculating the indexes of the
    hopping (sparse) matrix
    
    
    Parameters
    ----------

    n_sites : integer   number of lattice sites
    filling : integer   filling factor
    parity : string     even, odd or None
    
    """
    
    row = []
    column = []
    
    if parity == None:
        occu_list = [oo for oo in range(0,filling+1,1)]
    elif parity == 'even':
        occu_list = [oo for oo in range(0,filling+1,2)]
    elif parity == 'odd':
        occu_list = [oo for oo in range(1,filling+1,2)]
    else:
        print('Error: parity not recognized')
        
    o_sub_dim = 0
    for oo in occu_list:
        v_blocks = v_ocup_indx(n_sites,oo)
        for jj in range(n_sites-1):
    
            v_indx_neigh_r = np.where((v_blocks[::,jj] == 1) & (v_blocks[::,jj+1]== 0))
            v_indx_neigh_l = np.where((v_blocks[::,jj] == 0) & (v_blocks[::,jj+1]== 1))
            
            for mm, nn in itertools.product(v_indx_neigh_r[0],v_indx_neigh_l[0]):
    
                short_va = v_blocks[mm,::]
                short_vb = v_blocks[nn,::]
                short_va = np.delete(short_va,[jj,jj+1])
                short_vb = np.delete(short_vb,[jj,jj+1])
                
                diff = short_va - short_vb
                if np.dot(diff,diff) == 0:
                    row.append(nn+o_sub_dim)
                    column.append(mm+o_sub_dim)

        o_sub_dim = o_sub_dim + len(v_blocks)
        
    return row, column, o_sub_dim
    
def hopping_cav(t_hop,n_sites,filling,ph_n,ph_m,gg,parity=None,i_neigh=1):
    """ function to calculate the hopping in a lattice coupled to single mode photons """

    # INPUT
    # t_hop (float):      hopping parameter
    # n_sites (integer):  number of sites in the chain
    # filling (integer):  filling factor
    # ph_n (integer):     photon occupation of subspace
    # ph_m (integer):     photon occupation of subspace
    # gg (float):         light-matter coupling constant
    # parity (string):    even, odd or None
    # i_neigh (integer):  degree of neighbor, nearest neighbor (NN) or next NN
    # OUTPUT
    # result (sparse array):  hopping operator

    row = []
    column = []
    value = []

    if parity == None:
        occu_list = [oo for oo in range(0,filling+1,1)]
    elif parity == 'even':
        occu_list = [oo for oo in range(0,filling+1,2)]
    elif parity == 'odd':
        occu_list = [oo for oo in range(1,filling+1,2)]
    else:
        print('Error: parity not recognized')
        
    o_sub_dim = 0
    for oo in occu_list:
        v_blocks = v_ocup_indx(n_sites,oo)
        for jj in range(n_sites-i_neigh):
    
            v_indx_neigh_r = np.where((v_blocks[::,jj] == 1) & (v_blocks[::,jj+i_neigh]== 0))
            v_indx_neigh_l = np.where((v_blocks[::,jj] == 0) & (v_blocks[::,jj+i_neigh]== 1))
            #print(v_indx_neigh_r)
            #print(v_indx_neigh_l)
    
            for mm, nn in itertools.product(v_indx_neigh_r[0],v_indx_neigh_l[0]):
    
                short_va = v_blocks[mm,::]
                short_vb = v_blocks[nn,::]
                short_va = np.delete(short_va,[jj,jj+i_neigh])
                short_vb = np.delete(short_vb,[jj,jj+i_neigh])
                
                diff = short_va - short_vb
                if np.dot(diff,diff) == 0:
                    row.append(nn+o_sub_dim)
                    column.append(mm+o_sub_dim)
                    pol_in_gg = poly_J(ph_n,ph_m,gg)
                    t_dress = -t_hop*np.exp(-0.5*gg**2)*(-1j)**(abs(ph_m-ph_n))*pol_in_gg
                    value.append(t_dress)

        o_sub_dim = o_sub_dim + len(v_blocks)

    return coo_array((value,(row, column)), shape=(o_sub_dim,o_sub_dim))



def supergap(Delta,n_sites,filling,parity=None, j_neigh=1):
    """ function to calculate the superconducting gap (homogeneous) in a lattice """
    
    row = []
    column = []
    value = []
    
    if parity == None:
        occu_list = [oo for oo in range(0,filling-1,1)]
        skip = 1
        dim_count_a = 1
    elif parity == 'even':
        occu_list = [oo for oo in range(0,filling-1,2)]
        skip = 0
        dim_count_a = 0
    elif parity == 'odd':
        occu_list = [oo for oo in range(1,filling,2)]
        skip = 0
        dim_count_a = 0
    else:
        print('Error: parity not recognized')
        
    dim_count_b = 0
    for oo in occu_list:
        v_blocks = v_ocup_indx(n_sites,oo)
        w_blocks = v_ocup_indx(n_sites,oo+2)
        intermed_blck = math.comb(n_sites,oo+skip)
        
        for jj in range(n_sites-j_neigh):

            v_indx_neigh_blck_a = np.where((v_blocks[::,jj] == 0) & (v_blocks[::,jj+j_neigh]== 0))
            v_indx_neigh_blck_b = np.where((w_blocks[::,jj] == 1) & (w_blocks[::,jj+j_neigh]== 1))

            for mm, nn in itertools.product(v_indx_neigh_blck_a[0],v_indx_neigh_blck_b[0]):
    
                    short_va = v_blocks[mm,::]
                    short_wb = w_blocks[nn,::]
                    #print(oo,mm, short_va)
                    #print(oo+2,nn, short_wb)
                    short_va = np.delete(short_va,[jj,jj+j_neigh])
                    short_wb = np.delete(short_wb,[jj,jj+j_neigh])

                    diff = short_va - short_wb
                    if np.dot(diff,diff) == 0:
                        row.append(nn+dim_count_a+intermed_blck)
                        column.append(mm+dim_count_b)              
                        value.append(-Delta*(-1)**(v_blocks[mm,jj+1]))

        dim_count_b = dim_count_b + len(v_blocks)
        dim_count_a = dim_count_a + intermed_blck
        ##print(oo, dim_count_a, dim_count_b)

    if parity == None:
        if filling == n_sites:
            return coo_array((value,(row, column)), shape=(dim_count_a+1,dim_count_a+1))
        else:
            return coo_array((value,(row, column)), shape=(dim_count_a+len(w_blocks),dim_count_a+len(w_blocks))) 
    elif parity == 'even' or 'odd':
            return coo_array((value,(row, column)), shape=(dim_count_a+len(w_blocks),dim_count_a+len(w_blocks))) 

def super_indexes(n_sites,filling,parity):
    r""" function calculating the indexes of the
    superconducting (sparse) matrix
    
    
    Parameters
    ----------

    n_sites : integer   number of lattice sites
    filling : integer   filling factor
    parity : string     even, odd or None
    
    """
    
    row = []
    column = []
    j_site = []

    if parity == None:
        occu_list = [oo for oo in range(0,filling-1,1)]
        skip = 1
        dim_count_a = 1
    elif parity == 'even':
        occu_list = [oo for oo in range(0,filling-1,2)]
        skip = 0
        dim_count_a = 0
    elif parity == 'odd':
        occu_list = [oo for oo in range(1,filling,2)]
        skip = 0
        dim_count_a = 0
    else:
        print('Error: parity not recognized')
        
    dim_count_b = 0
    for oo in occu_list:
        v_blocks = v_ocup_indx(n_sites,oo)
        w_blocks = v_ocup_indx(n_sites,oo+2)
        intermed_blck = comb(n_sites,oo+skip)
        
        for jj in range(n_sites-1):

            v_indx_neigh_blck_a = np.where((v_blocks[::,jj] == 0) & (v_blocks[::,jj+1]== 0))
            v_indx_neigh_blck_b = np.where((w_blocks[::,jj] == 1) & (w_blocks[::,jj+1]== 1))

            for mm, nn in itertools.product(v_indx_neigh_blck_a[0],v_indx_neigh_blck_b[0]):
    
                    short_va = v_blocks[mm,::]
                    short_wb = w_blocks[nn,::]
                    #print(oo,mm, short_va)
                    #print(oo+2,nn, short_wb)
                    short_va = np.delete(short_va,[jj,jj+1])
                    short_wb = np.delete(short_wb,[jj,jj+1])

                    diff = short_va - short_wb
                    if np.dot(diff,diff) == 0:
                        row.append(nn+dim_count_a+intermed_blck)
                        column.append(mm+dim_count_b)
                        j_site.append(jj)

        dim_count_b = dim_count_b + len(v_blocks)
        dim_count_a = dim_count_a + intermed_blck

    if parity == None:
        if filling == n_sites:
            return row, column, j_site, dim_count_a+1
        else:
            return row, column, j_site, dim_count_a+len(w_blocks)
    elif parity == 'even' or 'odd':
            return row, column, j_site, dim_count_a+len(w_blocks)
        
def supergap_cav(Delta,n_sites,filling,ph_n,ph_m,gg,parity=None,j_neigh=1):
    """ function to calculate the superconducting gap in a lattice coupled to single mode photons """
    
    # To include NNN superconductivity in function supergap() the following changes are made:
    # in the loop for jj change the range(n_site-1) by range(n_site-i_neigh)
    # in v_indx_neigh_blck_a change jj + 1 by jj + i_neigh
    # in v_indx_neigh_blca_b change jj + 1 by jj + i_neigh
    # in short_va change jj+1 by jj + i_neigh
    # in short_vb change jj+1 by jj + i_neigh
    
    j0 = 0.5*(n_sites-1)     #conveninent if n_sites is even
    
    row = []
    column = []
    value = []
    
    if parity == None:
        occu_list = [oo for oo in range(0,filling-1,1)]
        skip = 1
        dim_count_a = 1
    elif parity == 'even':
        occu_list = [oo for oo in range(0,filling-1,2)]
        skip = 0
        dim_count_a = 0
    elif parity == 'odd':
        occu_list = [oo for oo in range(1,filling,2)]
        skip = 0
        dim_count_a = 0
    else:
        print('Error: parity not recognized')
        
    dim_count_b = 0
    for oo in occu_list:
        v_blocks = v_ocup_indx(n_sites,oo)
        w_blocks = v_ocup_indx(n_sites,oo+2)
        intermed_blck = comb(n_sites,oo+skip)
        
        for jj in range(n_sites-j_neigh):

            v_indx_neigh_blck_a = np.where((v_blocks[::,jj] == 0) & (v_blocks[::,jj+j_neigh]== 0))
            v_indx_neigh_blck_b = np.where((w_blocks[::,jj] == 1) & (w_blocks[::,jj+j_neigh]== 1))

            for mm, nn in itertools.product(v_indx_neigh_blck_a[0],v_indx_neigh_blck_b[0]):
    
                    short_va = v_blocks[mm,::]
                    short_wb = w_blocks[nn,::]
                    #print(oo,mm, short_va)
                    #print(oo+2,nn, short_wb)
                    short_va = np.delete(short_va,[jj,jj+j_neigh])
                    short_wb = np.delete(short_wb,[jj,jj+j_neigh])

                    diff = short_va - short_wb
                    if np.dot(diff,diff) == 0:
                        row.append(nn+dim_count_a+intermed_blck)
                        column.append(mm+dim_count_b)
                        #odd_indx = 2*(jj-j0)+1
                        argum = gg*(2*(jj-j0)+1)
                        pol_in_arg = poly_J(ph_n,ph_m,argum)
                        Delta_dress = -Delta*np.exp(-0.5*argum**2)*(-1j)**(abs(ph_m-ph_n))*pol_in_arg 
                        value.append(Delta_dress)

        dim_count_b = dim_count_b + len(v_blocks)
        dim_count_a = dim_count_a + intermed_blck
        ##print(oo, dim_count_a, dim_count_b)

    if parity == None:
        if filling == n_sites:
            return coo_array((value,(row, column)), shape=(dim_count_a+1,dim_count_a+1))
        else:
            return coo_array((value,(row, column)), shape=(dim_count_a+len(w_blocks),dim_count_a+len(w_blocks))) 
    elif parity == 'even' or 'odd':
            return coo_array((value,(row, column)), shape=(dim_count_a+len(w_blocks),dim_count_a+len(w_blocks))) 

def onsite(chem, n_sites, filling, parity=None):
    """ function to calculate the onsite (chemical) potential """

    row = []
    value = []
    
    if parity == None:
        occu_list = [oo for oo in range(0,filling+1,1)]
    elif parity == 'even':
        occu_list = [oo for oo in range(0,filling+1,2)]
    elif parity == 'odd':
        occu_list = [oo for oo in range(1,filling+1,2)]
    else:
        print('Error: parity not recognized')
        
    o_sub_dim = 0
    for oo in occu_list:    
        v_blocks = v_ocup_indx(n_sites,oo)

        for mm in range(len(v_blocks)):

            v_indx = np.where(v_blocks[mm,::]==1)
            row.append(mm+o_sub_dim)
            ons_terms = [-chem for jj in v_indx[0][::]]
            value.append(np.sum(np.array(ons_terms))+0.5*chem*n_sites)
            
        o_sub_dim = o_sub_dim + len(v_blocks)
        
    return coo_array((value,(row,row)), shape=(o_sub_dim,o_sub_dim))
              
def g_coup_terms_depre(tt,D0,gg,par,n_sites,N_ph_max):
    """ function to calculate the sum of all the light-matter coupled terms (deprecated) """
    
    gg = gg/np.sqrt(n_sites)
    
    sub_blocks = []
    for n_ph, m_ph in itertools.product(range(N_ph_max),range(N_ph_max)):
    
        LD_T = hopping_cav(tt, n_sites, n_sites, n_ph, m_ph, gg, par)
        TT = LD_T + LD_T.conjugate().T
        
        LD_D = supergap_cav(D0, n_sites, n_sites, n_ph, m_ph, gg, par)
        DD = LD_D +  LD_D.conjugate().T
        
        sub_blocks.append(TT+DD)
    
    row_blocks = []
    for n_ph in range(N_ph_max):
        sub = sub_blocks[N_ph_max*n_ph]
        for m_ph in range(1,N_ph_max):
            sub = bmat([[sub,sub_blocks[N_ph_max*n_ph+m_ph]]])
        row_blocks.append(sub)

    T_pl_D = row_blocks[0]
    for n_ph in range(1,N_ph_max):
        T_pl_D = bmat([[T_pl_D],[row_blocks[n_ph]]])

    return T_pl_D
    
def g_coup_terms_deprecated(tt,D0,gg,par,n_sites,N_ph_max,N_ph_cut=5):
    """ function to calculate the sum of all the diagonal light-matter terms """
    
    gg = gg/np.sqrt(n_sites)
    
    spa_blocks = []
    for n_ph in range(N_ph_max):
        rows = []
        for m_ph in range(N_ph_max):
            if abs(m_ph - n_ph) < N_ph_cut:
                #print('n:',n_ph,', m:',m_ph)
                LD_T = hopping_cav(tt, n_sites, n_sites, n_ph, m_ph, gg, par)
                #MixD_T = hopp_eff(tt,n_sites,n_ph,m_ph,gg,par)
                #TT = MixD_T + MixD_T.conjugate().T
                TT = LD_T + LD_T.conjugate().T
                
                LD_D = supergap_cav(D0, n_sites, n_sites, n_ph, m_ph, gg, par)
                #LD_D = supgap_eff(D0,n_sites,n_ph,m_ph,gg,par)
                DD = LD_D +  LD_D.conjugate().T

                rows.append(TT+DD)
            else:
                rows.append(None)
        spa_blocks.append(rows)

    return bmat(spa_blocks)

def g_coup_terms(tt,D0,gg,par,n_sites,N_ph_max,N_ph_cut=5):
    r""" function calculating the sum of the diagonal light-matter terms
    This function is an update of the functions:
    g_coup_terms_deprecated()
    g_coup_terms_depre()
    
    Parameters
    ----------

    tt : float          hopping parameter
    D0 : float          superconducting gap parameter
    gg : float          light-matter coupling strength
    par : string        even, odd or None
    n_sites : integer   number of lattice sites
    N_ph_max : integer  cutoff on the number of photon Fock states
    N_ph_cut : integer  cutoff on the number of photon transitions
    
    """
    gg = gg/np.sqrt(n_sites)
    j0 = 0.5*(n_sites-1)     #conveninent if n_sites is even
        
    hop_row, hop_column, hop_dim = hopping_indexes(n_sites,n_sites,par)
    sup_row, sup_column, list_j, sup_dim = super_indexes(n_sites,n_sites,par)
    
    spa_blocks = []
    for n_ph in range(N_ph_max):
        rows = []
        for m_ph in range(N_ph_max):
            if abs(m_ph - n_ph) < N_ph_cut:
                #print('n:',n_ph,', m:',m_ph)
                #LD_T = hopping_cav(tt, n_sites, n_sites, n_ph, m_ph, gg, par)
                
                pol_in_gg = poly_J(n_ph,m_ph,gg)
                t_dress = -tt*np.exp(-0.5*gg**2)*(-1j)**(abs(m_ph-n_ph))*pol_in_gg
                value = [t_dress]*(len(hop_row))

                LD_T = coo_array((value,(hop_row, hop_column)), shape=(hop_dim,hop_dim))
    
                #MixD_T = hopp_eff(tt,n_sites,n_ph,m_ph,gg,par)
                #TT = MixD_T + MixD_T.conjugate().T
                TT = LD_T + LD_T.conjugate().T
                
                #LD_D = supergap_cav(D0, n_sites, n_sites, n_ph, m_ph, gg, par)
                argum = [gg*(2*(jj-j0)+1) for jj in list_j]
                pol_in_arg = [poly_J(n_ph,m_ph,ph_phase) for ph_phase in argum]
                Delta_dress = [-D0*np.exp(-0.5*argum[ll]**2)*(-1j)**(abs(m_ph-n_ph))*pol_in_arg[ll] for ll in range(len(argum))]

                LD_D = coo_array((Delta_dress,(sup_row, sup_column)), shape=(sup_dim,sup_dim))

                #LD_D = supgap_eff(D0,n_sites,n_ph,m_ph,gg,par)
                DD = LD_D +  LD_D.conjugate().T

                rows.append(TT+DD)
            else:
                rows.append(None)
        spa_blocks.append(rows)

    return bmat(spa_blocks)

def g_uncoup_terms(mu,omg,par,n_sites,N_ph_max):
    """ function to calculate the sum of all the light-matter coupled terms """
    
    ext_onsite = onsite(mu, n_sites, n_sites, par)
    ph_diag = 0*eye(int(2**(n_sites-1)))
    for n_ph in range(1,N_ph_max):
        ext_onsite = bmat([[ext_onsite,None],[None,onsite(mu, n_sites, n_sites, par)]])
        ph_diag = bmat([[ph_diag,None],[None,n_ph*omg*eye(int(2**(n_sites-1)))]])

    return ext_onsite + ph_diag

def statistics():
    """ function to check the time and memory comsumption of the functions
    building the hamiltonian """
    
    n_sites =14
    fill = 14
    parlist = ['even','odd']
    memlist = []
    sparselist = []
    
    for ns in range(2,14):
        LD_T = hopping(1.0, ns, ns, parlist[0])
        TT = LD_T + LD_T.conjugate().T
        
        LD_D = supergap(0.3, ns, ns, parlist[0])
        DD = LD_D +  LD_D.conjugate().T
        
        UU = onsite(0.8, ns, ns, parlist[0])

        ham = UU + TT + DD
        
        nzero = ham.count_nonzero()
        print(nzero)
        memlist.append(nzero*2*8)
        densedim = 2**ns * 2**ns
        sparselist.append((densedim - 2*nzero)/densedim)
        
    fig, ax = plt.subplots(1,2,figsize=(10,4.5))
    
    ax[0].plot(np.arange(2,14,1),np.array(memlist)/1000, '-o')
    ax[1].plot(np.arange(2,14,1),np.array(sparselist), '-o')
    ax[0].set_ylabel('memory [MBytes]')
    ax[0].set_xlabel('number of sites')
    ax[1].set_ylabel('sparsity = (dim(dense)-dim(sparse)/dim(dense)')
    ax[1].set_xlabel('number of sites')
    
    plt.show()
    plt.savefig('matrix_statistics.pdf')
    plt.close()
    
def E_nocav(chemstr,chem,n_sit):
    """ function to calculate the energy spectrum of the Kitaev lattice
    without coupling to photons """
    
    parlist = ['even','odd']

    fl_name = 'EVecs_'+chemstr+str(chem)+'_ns_'+str(n_sit)+'.hdf5'

    with h5py.File(fl_name, 'w') as f: 
        f.create_dataset(chemstr, data=chem)
                    

    for par in parlist:
        
        hop_row, hop_column, hop_dim = hopping_indexes(n_sit,n_sit,par)
        sup_row, sup_column, list_j, sup_dim = super_indexes(n_sit,n_sit,par)

        #Elist = []
        
        #MixD_T = hopp_eff(1.0,n_sit,0,0,0.0,par)
        #MixD_T = hopping_cav(1.0, n_sit, n_sit, 0, 0, 0, par)
        value = [-1.0]*(len(hop_row))
        LD_T = coo_array((value,(hop_row, hop_column)), shape=(hop_dim,hop_dim))

        TT = LD_T + LD_T.conjugate().T

        #LD_D = supgap_eff(0.2,n_sit,0,0,0.0,par)
        #LD_D = supergap_cav(0.2, n_sit, n_sit, 0, 0, 0, par)
        Delta_values = [-0.2]*len(sup_row)
        LD_D = coo_array((Delta_values,(sup_row, sup_column)), shape=(sup_dim,sup_dim))
        
        DD = LD_D +  LD_D.conjugate().T

        #for mu in muarr:
            
        UU = onsite(chem, n_sit, n_sit, par)

        ham = UU + TT + DD
    
        if n_sit < 8:
            jam = ham.toarray()
            Evals, Evecs = eigh(jam, eigvals_only=False)
        else:
            Evals, Evecs = eigsh(ham, k=50, which='SA', return_eigenvectors=True)
    
        sort_indx = np.argsort(Evals)
        Evals = Evals[sort_indx]
        Evecs = Evecs[::,sort_indx]


        print(Evals.shape)
            
        with h5py.File(fl_name, 'a') as f:
            f.create_dataset('E_mat_par_'+par, data=Evals)
            f.create_dataset('V_mat_par_'+par, data=Evecs[::,0:10:])

    
def E_Vecs_vs_params(varstr,array,par1str,par1,par2str,par2,n_sit,nph_cut):
    """ function to calculate the energy spectrum of the Kitaev chain coupled to
    single mode photons """
    # INPUT
    # varstr (string): variable that labels the input array
    # array (float array): array of input values
    # par1str (string): variable that labels one parameter of the system
    # par1 (float):  value for the parameter of the system with label par1str
    # par2str (string): variable that labels another parameter of the system
    # par2 (float):  value for the parameter of the system with label par2str
    # n_sit (integer): number of sites
    # nph_cut (integer): number of photons cutoff
    
    parlist = ['even','odd']
    
    fl_name = 'EVecs_vs_'+varstr+'_'+par1str+str(par1)+'_'+par2str+str(par2)+'_ns_'+str(n_sit)+'.hdf5'

    with h5py.File(fl_name, 'w') as f:
        f.create_dataset(varstr, data=array)
        
    #N_phot = ph_numb_op(n_sit,nph_cut)
    
    for par in parlist:
    
        Elist = []
        lowVs = []
        av_nph = []
        start = time.time()
        H_elight = g_coup_terms(1.0,0.2,par2,par,n_sit,nph_cut,N_ph_cut=5)
        end = time.time()
        print('exe time coupled terms: ',end-start)
        
        for param in array:
        
                if varstr == 'mu':
                    H_bdiag = g_uncoup_terms(param,par1,par,n_sit,nph_cut)      #g_uncoup_terms(mu,omg,par,n_sites,N_ph_max)
                elif varstr == 'w':
                    H_bdiag = g_uncoup_terms(par1,param,par,n_sit,nph_cut)      #g_uncoup_terms(mu,omg,par,n_sites,N_ph_max)
                else:
                    print('option not recognized in param (mu,w)')
        
                ham = H_elight + H_bdiag
                
                if n_sit < 7:
                    jam = ham.toarray()
                    Evals, Evecs = eigh(jam, eigvals_only=False)
                else:
                    Evals, Evecs = eigsh(ham, k=50, which='SA', return_eigenvectors=True)
                    
                sort_indx = np.argsort(Evals)
                Evals = Evals[sort_indx]
                Evecs = Evecs[::,sort_indx]
                Elist.append(Evals)
                lowVs.append(Evecs[::,0:10:])
                print(par,param)
                #print('Numb. of Es:',Evals.shape,'Numb. of Es & V dim:',Evecs.shape)
                
    
        Emtx = np.array(Elist)
        Vmtx = np.array(lowVs)
        #print(Vmtx.shape, N_phot.shape)

        if n_sit < 7:
            Emtx = Emtx.reshape((len(array),2**(n_sit-1)*nph_cut))
        else:
            Emtx = Emtx.reshape((len(array),50))
            
        with h5py.File(fl_name, 'a') as f:
            f.create_dataset('E_mat_par_'+par, data=Emtx)
            f.create_dataset('V_mat_par_'+par, data=Vmtx)
            
def mu_crit(n):
    """ function to calculate the critical values of mu where the
    parity switches """

    if n % 2 == 0:
        muroots = [2*np.sqrt(1-0.2**2)*np.cos(np.pi*pp/(n+1)) for pp in range(1,int(0.5*n)+1)]
        muroots.append(-muroots[-1])
    else:
        muroots = [2*np.sqrt(1-0.2**2)*np.cos(np.pi*pp/(n+1)) for pp in range(1,int(0.5*(n-1))+1)]
        muroots.append(0.0)
        
    return muroots

def irr_domu(ns):
    """ function to calculate an unevenly space input array denser around
    the critical mu's where the parity switches """

    muclist = mu_crit(ns)
    d_muc_l = muclist[0]-muclist[1]

    intvalgap = np.linspace(3.0,muclist[0]+0.2*d_muc_l,9,endpoint=False)
    intvalgcl = np.linspace(muclist[0]+0.2*d_muc_l,muclist[0]-0.15*d_muc_l,7,endpoint=False)
    
    arrir = np.concatenate((intvalgap,intvalgcl))
    for ii in range(1,len(muclist)-1):

        d_muc_r = muclist[ii-1]-muclist[ii]
        d_muc_l = muclist[ii]-muclist[ii+1]
        
        arrspar = np.linspace(muclist[ii]+0.75*d_muc_r,muclist[ii]+0.15*d_muc_r,4+ii,endpoint=False)
        arrir = np.concatenate((arrir,arrspar), axis=0)
        arrdens = np.linspace(muclist[ii]+0.15*d_muc_r, muclist[ii]-0.15*d_muc_l,6+ii,endpoint=False)
        arrir = np.concatenate((arrir,arrdens), axis=0)
        
    intvalar0 = np.linspace(muclist[-2]-0.15*d_muc_l, 1e-2,13)
    
    arrir = np.concatenate((arrir,intvalar0))
    
    return arrir

def main():
    r""" the main function 
    
    
    Parameters
    ----------

    ns : integer         number of lattice sites
    nph : integer        cutoff in the number of photons
    mu_chem: float       the chemical potential, \mu
    w_omega: float       the cavity frequency, \omega
    g_coupling: float    the light-matter coupling strength 

    comment:             the other two physical parametes (hopping 
                         and superconducting gap) are set in functions
                         E_nocav(,,,) and E_Vecs_vs_params(,,,,)
    
    """

    
    ns =11
    nph =21
    mu_chem = 1.5        
    w_omega = 0.8
    g_coupling = 0.25

    # diagonalization of the matter hamiltonian (Kitaev chain) at one value of the chemical potential
    #E_nocav('mu',mu_chem,ns)
    
    intval = np.linspace(7e-3,3.0,49)
    # diagonalization of the light-matter hamiltonian as a function of frequency (w)
    E_Vecs_vs_params('w',intval,'mu',mu_chem,'g',g_coupling,ns,nph)     

    # diagonalization of the light-matter hamiltonian as a function of chemical potential (mu)    
    intval_irr = irr_domu(ns)
    #E_Vecs_vs_params('mu',intval_irr,'w',w_omega,'g',g_coupling,ns,nph) # E vs mu "irregular domain in mu"


if __name__ == '__main__':
    main()
