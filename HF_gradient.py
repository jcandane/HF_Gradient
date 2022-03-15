import numpy as np 

π = np.pi 

def h_deriv(atom_id, h1, mol):
    shl0, shl1, p0, p1 = (mol.aoslice_by_atom())[atom_id]
    with mol.with_rinv_at_nucleus(atom_id):
        vrinv  = (-mol.atom_charge(atom_id)) * mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
        vrinv += mol.intor('ECPscalar_iprinv', comp=3)
    vrinv[:,p0:p1] += h1[:,p0:p1]
    return vrinv + vrinv.swapaxes(1,2) 

def S_deriv(atom_id, S_xAB, mol):
    shl0, shl1, p0, p1 = (mol.aoslice_by_atom())[atom_id]

    vrinv = np.zeros(S_xAB.shape)
    vrinv[:, p0:p1, :] += S_xAB[:, p0:p1, :]
    
    return vrinv + vrinv.swapaxes(1,2)

def I_deriv(atom_id, I_xABCD, mol):
    shl0, shl1, p0, p1 = (mol.aoslice_by_atom())[atom_id]

    vrinv  = np.zeros(I_xABCD.shape)
    vrinv[:, p0:p1, :, :, :] += I_xABCD[:, p0:p1, :, :, :]
    
    vrinv += np.einsum("xABCD -> xCDAB", vrinv) 
    vrinv += np.einsum("xABCD -> xBACD", vrinv) 
    vrinv += np.einsum("xABCD -> xABDC", vrinv)
    vrinv += np.einsum("xABCD -> xBADC", vrinv)
    
    return vrinv/4.

def get_f_ix(mol, DA, DB, FA, FB):
    """
    GIVEN:
    mol : PySCF mol object
    DA, DB: AO Density Matrices (2d numpy arrays)
    FA, FB: AO Fock Matrices (2d numpy arrays)
    GET:
    HF force between atoms
    """
    h_xAB    = -mol.intor('ECPscalar_ipnuc', comp=3)
    h_xAB   += -mol.intor('int1e_ipkin', comp=3)
    h_xAB   += -mol.intor('int1e_ipnuc', comp=3)
    S_xAB    = -mol.intor('int1e_ipovlp', comp=3)
    I_xABCD  = -mol.intor('int2e_ip1', comp=3)
    ff_ix    = np.zeros( (len(mol.aoslice_by_atom()), 3) )
    for i in range(len(mol.aoslice_by_atom())):
        dI_x = I_deriv(i, I_xABCD, mol)
        dS_x = S_deriv(i, S_xAB, mol)
        ff_ix[i]  = -1.0*np.einsum("mn, Xmn -> X",  DA+DB, h_deriv(i, h_xAB, mol))
        ff_ix[i] -=  0.5*np.einsum("nm, ls, Xmnls -> X", DA+DB, DA+DB, dI_x, optimize=True) ## dJ
        ff_ix[i] +=  0.5*np.einsum("nm, ls, Xmlsn -> X", DA, DA, dI_x, optimize=True) ## dKα
        ff_ix[i] +=  0.5*np.einsum("nm, ls, Xmlsn -> X", DB, DB, dI_x, optimize=True) ## dKβ

        ## Pulay
        ff_ix[i] += 1.0*np.einsum("ij, Xjk, kl, il -> X", DA, dS_x, DA, FA, optimize=True)
        ff_ix[i] += 1.0*np.einsum("ij, Xjk, kl, il -> X", DB, dS_x, DB, FB, optimize=True)

    return ff_ix

def nuclei_force(Z, R_ix):
    """GIVEN: atomic number, Z, & positions R_ix calculate force """
    R_ijx = R_ix[None, :,:] - R_ix[:, None,:]
    f_ij  = Z[:, None] * Z[None,:] /(np.linalg.norm(R_ijx, axis=2)**3 + np.eye(len(R_ix)) ) 
    f_ij *= (1-np.eye(len(Z)))
    return -np.einsum("ij, ijx -> ix", f_ij, R_ijx)


