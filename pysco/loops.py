"""
This module defines looping patterns within PySCo.
"""
import numpy as np
import numpy.typing as npt
from numba import config, njit, prange

@njit(fastmath=False, parallel=True)
def ravel_rhs_3f(x, b, func, float1, float2, float3):
    x_ravel = x.ravel()
    b_ravel = b.ravel()
    for i in prange(len(x_ravel)):
        func(x_ravel, b_ravel, i, float1, float2, float3)


# Offset is the maximum stencil neighbour.
# x[i+1] - x[i-1] : offset = 1
# x[i+2] + x[i+1] - x[i-1] - x[i-2] : offset = 2

@njit(fastmath=False, parallel=True)
def offset_1f(out, x, func, float1, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, i, j, k, float1)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset,offset):
                func(out, x, i, j, k, float1)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset,offset):
                func(out, x, i, j, k, float1)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset,offset):
                func(out, x, i, j, k, float1)
    # Edges
    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in prange(offset, N-offset):
                func(out, x, i, j, k, float1)
    for i in range(-offset,offset):
        for k in range(-offset,offset):
            for j in prange(offset, N-offset):
                func(out, x, i, j, k, float1)
    for j in range(-offset,offset):
        for k in range(-offset,offset):
            for i in prange(offset, N-offset):
                func(out, x, i, j, k, float1)
    # Corners
    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in range(-offset,offset):
                func(out, x, i, j, k, float1)

@njit(fastmath=False, parallel=True)
def offset_2f(out, x, func, float1, float2, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, i, j, k, float1, float2)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset,offset):
                func(out, x, i, j, k, float1, float2)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset,offset):
                func(out, x, i, j, k, float1, float2)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset,offset):
                func(out, x, i, j, k, float1, float2)
    # Edges
    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2)
    for i in range(-offset,offset):
        for k in range(-offset,offset):
            for j in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2)
    for j in range(-offset,offset):
        for k in range(-offset,offset):
            for i in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2)
    # Corners
    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in range(-offset,offset):
                func(out, x, i, j, k, float1, float2)

@njit(fastmath=False, parallel=True)
def offset_3f(out, x, func, float1, float2, float3, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, i, j, k, float1, float2, float3)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset,offset):
                func(out, x, i, j, k, float1, float2, float3)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset,offset):
                func(out, x, i, j, k, float1, float2, float3)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset,offset):
                func(out, x, i, j, k, float1, float2, float3)
    # Edges
    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2, float3)
    for i in range(-offset,offset):
        for k in range(-offset,offset):
            for j in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2, float3)
    for j in range(-offset,offset):
        for k in range(-offset,offset):
            for i in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2, float3)
    # Corners
    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in range(-offset,offset):
                func(out, x, i, j, k, float1, float2, float3)

@njit(fastmath=False, parallel=True)
def offset_4f(out, x, func, float1, float2, float3, float4, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, i, j, k, float1, float2, float3, float4)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset,offset):
                func(out, x, i, j, k, float1, float2, float3, float4)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset,offset):
                func(out, x, i, j, k, float1, float2, float3, float4)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset,offset):
                func(out, x, i, j, k, float1, float2, float3, float4)
    # Edges
    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2, float3, float4)
    for i in range(-offset,offset):
        for k in range(-offset,offset):
            for j in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2, float3, float4)
    for j in range(-offset,offset):
        for k in range(-offset,offset):
            for i in prange(offset, N-offset):
                func(out, x, i, j, k, float1, float2, float3, float4)
    # Corners
    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in range(-offset,offset):
                func(out, x, i, j, k, float1, float2, float3, float4)


@njit(fastmath=False, parallel=True)
def offset_rhs_2f(out, x, b, func, float1, float2, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2)
    # Edges
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2)
    for i in range(-offset, offset):
        for k in range(-offset, offset):
            for j in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2)
    for j in range(-offset, offset):
        for k in range(-offset, offset):
            for i in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2)
    # Corners
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2)

@njit(fastmath=False, parallel=True)
def offset_rhs_3f(out, x, b, func, float1, float2, float3, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2, float3)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2, float3)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2, float3)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2, float3)
    # Edges
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2, float3)
    for i in range(-offset, offset):
        for k in range(-offset, offset):
            for j in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2, float3)
    for j in range(-offset, offset):
        for k in range(-offset, offset):
            for i in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2, float3)
    # Corners
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2, float3)

@njit(fastmath=False, parallel=True)
def offset_rhs_4f(out, x, b, func, float1, float2, float3, float4, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2, float3, float4)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2, float3, float4)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2, float3, float4)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2, float3, float4)
    # Edges
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2, float3, float4)
    for i in range(-offset, offset):
        for k in range(-offset, offset):
            for j in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2, float3, float4)
    for j in range(-offset, offset):
        for k in range(-offset, offset):
            for i in prange(offset, N-offset):
                func(out, x, b, i, j, k, float1, float2, float3, float4)
    # Corners
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in range(-offset, offset):
                func(out, x, b, i, j, k, float1, float2, float3, float4)

@njit(fastmath=False, parallel=True)
def offset_2rhs_2f(out, x, b, rhs, func, float1, float2, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, b, rhs, i, j, k, float1, float2)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset, offset):
                func(out, x, b, rhs, i, j, k, float1, float2)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset, offset):
                func(out, x, b, rhs, i, j, k, float1, float2)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset, offset):
                func(out, x, b, rhs, i, j, k, float1, float2)
    # Edges
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in prange(offset, N-offset):
                func(out, x, b, rhs, i, j, k, float1, float2)
    for i in range(-offset, offset):
        for k in range(-offset, offset):
            for j in prange(offset, N-offset):
                func(out, x, b, rhs, i, j, k, float1, float2)
    for j in range(-offset, offset):
        for k in range(-offset, offset):
            for i in prange(offset, N-offset):
                func(out, x, b, rhs, i, j, k, float1, float2)
    # Corners
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in range(-offset, offset):
                func(out, x, b, rhs, i, j, k, float1, float2)

@njit(fastmath=False, parallel=True)
def offset_2rhs_3f(out, x, b, rhs, func, float1, float2, float3, offset):
    N = out.shape[0] 
    # Interior
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(offset, N-offset):
                func(out, x, b, rhs, i, j, k, float1, float2, float3)

    if offset == 0:
        return
    
    # Faces
    for j in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for i in range(-offset, offset):
                func(out, x, b, rhs, i, j, k, float1, float2, float3)
    for i in prange(offset, N-offset):
        for k in range(offset, N-offset):
            for j in range(-offset, offset):
                func(out, x, b, rhs, i, j, k, float1, float2, float3)
    for i in prange(offset, N-offset):
        for j in range(offset, N-offset):
            for k in range(-offset, offset):
                func(out, x, b, rhs, i, j, k, float1, float2, float3)
    # Edges
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in prange(offset, N-offset):
                func(out, x, b, rhs, i, j, k, float1, float2, float3)
    for i in range(-offset, offset):
        for k in range(-offset, offset):
            for j in prange(offset, N-offset):
                func(out, x, b, rhs, i, j, k, float1, float2, float3)
    for j in range(-offset, offset):
        for k in range(-offset, offset):
            for i in prange(offset, N-offset):
                func(out, x, b, rhs, i, j, k, float1, float2, float3)
    # Corners
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            for k in range(-offset, offset):
                func(out, x, b, rhs, i, j, k, float1, float2, float3)

    

@njit(fastmath=False, parallel=True)
def gauss_seidel_3f(x, b, func, float1, float2, float3):
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    n = x.shape[0]
    N = n // 2
    # Red 
    for i in prange(x.shape[0] >> 1):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, iim1, jjm1, kkm1, float1, float2, float3)
                func(x, b, iim1, jj, kk, float1, float2, float3)
                func(x, b, ii, jjm1, kk, float1, float2, float3)
                func(x, b, ii, jj, kkm1, float1, float2, float3)
    # Black
    for i in prange(N):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, iim1, jjm1, kk, float1, float2, float3)
                func(x, b, iim1, jj, kkm1, float1, float2, float3)
                func(x, b, ii, jjm1, kkm1, float1, float2, float3)
                func(x, b, ii, jj, kk, float1, float2, float3)


@njit(fastmath=False, parallel=True)
def gauss_seidel_4f(x, b, func, float1, float2, float3, float4):
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    n = x.shape[0]
    N = n // 2
    # Red 
    for i in prange(x.shape[0] >> 1):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, iim1, jjm1, kkm1, float1, float2, float3, float4)
                func(x, b, iim1, jj, kk, float1, float2, float3, float4)
                func(x, b, ii, jjm1, kk, float1, float2, float3, float4)
                func(x, b, ii, jj, kkm1, float1, float2, float3, float4)
    # Black
    for i in prange(N):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, iim1, jjm1, kk, float1, float2, float3, float4)
                func(x, b, iim1, jj, kkm1, float1, float2, float3, float4)
                func(x, b, ii, jjm1, kkm1, float1, float2, float3, float4)
                func(x, b, ii, jj, kk, float1, float2, float3, float4)


@njit(fastmath=False, parallel=True)
def gauss_seidel_rhs_3f(x, b, rhs, func, float1, float2, float3):
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    n = x.shape[0]
    N = n // 2
    # Red 
    for i in prange(x.shape[0] >> 1):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, rhs, iim1, jjm1, kkm1, float1, float2, float3)
                func(x, b, rhs, iim1, jj, kk, float1, float2, float3)
                func(x, b, rhs, ii, jjm1, kk, float1, float2, float3)
                func(x, b, rhs, ii, jj, kkm1, float1, float2, float3)
    # Black
    for i in prange(N):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, rhs, iim1, jjm1, kk, float1, float2, float3)
                func(x, b, rhs, iim1, jj, kkm1, float1, float2, float3)
                func(x, b, rhs, ii, jjm1, kkm1, float1, float2, float3)
                func(x, b, rhs, ii, jj, kk, float1, float2, float3)


@njit(fastmath=False, parallel=True)
def gauss_seidel_rhs_4f(x, b, rhs, func, float1, float2, float3, float4):
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    n = x.shape[0]
    N = n // 2
    # Red 
    for i in prange(x.shape[0] >> 1):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, rhs, iim1, jjm1, kkm1, float1, float2, float3, float4)
                func(x, b, rhs, iim1, jj, kk, float1, float2, float3, float4)
                func(x, b, rhs, ii, jjm1, kk, float1, float2, float3, float4)
                func(x, b, rhs, ii, jj, kkm1, float1, float2, float3, float4)
    # Black
    for i in prange(N):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, rhs, iim1, jjm1, kk, float1, float2, float3, float4)
                func(x, b, rhs, iim1, jj, kkm1, float1, float2, float3, float4)
                func(x, b, rhs, ii, jjm1, kkm1, float1, float2, float3, float4)
                func(x, b, rhs, ii, jj, kk, float1, float2, float3, float4)


@njit(fastmath=False, parallel=True)
def gauss_seidel_rhs_5f(x, b, rhs, func, float1, float2, float3, float4, float5):
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    n = x.shape[0]
    N = n // 2
    # Red 
    for i in prange(x.shape[0] >> 1):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, rhs, iim1, jjm1, kkm1, float1, float2, float3, float4, float5)
                func(x, b, rhs, iim1, jj, kk, float1, float2, float3, float4, float5)
                func(x, b, rhs, ii, jjm1, kk, float1, float2, float3, float4, float5)
                func(x, b, rhs, ii, jj, kkm1, float1, float2, float3, float4, float5)
    # Black
    for i in prange(N):
        ii = 2*i
        iim1 = ii - 1
        for j in range(N):
            jj = 2*j
            jjm1 = jj - 1
            for k in range(N):
                kk = 2*k
                kkm1 = kk - 1
                func(x, b, rhs, iim1, jjm1, kk, float1, float2, float3, float4, float5)
                func(x, b, rhs, iim1, jj, kkm1, float1, float2, float3, float4, float5)
                func(x, b, rhs, ii, jjm1, kkm1, float1, float2, float3, float4, float5)
                func(x, b, rhs, ii, jj, kk, float1, float2, float3, float4, float5)