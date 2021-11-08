#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange_fourth_order.go for more details.

///////////////////////////////////////////////////
// Important note: Currently only works for Nz=1 //
///////////////////////////////////////////////////

extern "C" __global__ void
addexchangefourthorder(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ aSecondOrderLUT2d, float* __restrict__ aFourthOrderLUT2d,
            uint8_t* __restrict__ regions,
            float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int    I  = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    uint8_t r0 = regions[I];
    float3  B  = make_float3(0.0,0.0,0.0);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float aSecondOrder__;  // second-order exchange stiffness
    float aFourthOrder__;  // fourth-order exchange stiffness


    //////////////////
    // Central Spin //
    //////////////////
    i_              = idx(ix, iy, iz);
    m_              = make_float3(mx[i_], my[i_], mz[i_]);                        // load m
    m_              = ( is0(m_)? m0: m_ );                                        // replace missing non-boundary neighbor
    aSecondOrder__  = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= 4 * aSecondOrder__ * (1/(cx*cx) + 1/(cy*cy)) * m_;
    B              -= 12 * aFourthOrder__ * (1/(cx*cx*cx*cx) + 1/(cy*cy*cy*cy)) * m_;
    B              -= 16 * aFourthOrder__ / (cx*cx*cy*cy) * m_;


    ///////////////////////////////
    // Direct Nearest Neighbours //
    ///////////////////////////////
    
    // Left neighbour
    i_              = idx(lclampx(ix-1), iy, iz);                  // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aSecondOrder__  = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              += (2 * aSecondOrder__ / (cx*cx)) * m_;
    B              += (8 * aFourthOrder__ / (cx*cx*cx*cx)) * m_;
    B              += (8 * aFourthOrder__ / (cx*cx*cy*cy)) * m_;

    // Right neighbour
    i_              = idx(hclampx(ix+1), iy, iz);                  // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aSecondOrder__  = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              += (2 * aSecondOrder__ / (cx*cx)) * m_;
    B              += (8 * aFourthOrder__ / (cx*cx*cx*cx)) * m_;
    B              += (8 * aFourthOrder__ / (cx*cx*cy*cy)) * m_;

    // Below neighbour
    i_              = idx(ix, lclampy(iy-1), iz);                  // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aSecondOrder__  = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              += (2 * aSecondOrder__ / (cy*cy)) * m_;
    B              += (8 * aFourthOrder__ / (cy*cy*cy*cy)) * m_;
    B              += (8 * aFourthOrder__ / (cx*cx*cy*cy)) * m_;

    // Above neighbour
    i_              = idx(ix, hclampy(iy+1), iz);                  // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aSecondOrder__  = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              += (2 * aSecondOrder__ / (cy*cy)) * m_;
    B              += (8 * aFourthOrder__ / (cy*cy*cy*cy)) * m_;
    B              += (8 * aFourthOrder__ / (cx*cx*cy*cy)) * m_;


    /////////////////////////////////
    // Diagonal Nearest Neighbours //
    /////////////////////////////////

    // Bottom-left neighbour
    i_              = idx(lclampx(ix-1), lclampy(iy-1), iz);       // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= (4 * aFourthOrder__ / (cx*cx*cy*cy)) * m_;

    // Top-left neighbour
    i_              = idx(lclampx(ix-1), hclampy(iy+1), iz);       // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= (4 * aFourthOrder__ / (cx*cx*cy*cy)) * m_;

    // Bottom-right neighbour
    i_              = idx(hclampx(ix+1), lclampy(iy-1), iz);       // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= (4 * aFourthOrder__ / (cx*cx*cy*cy)) * m_;

    // Top-right neighbour
    i_              = idx(hclampx(ix+1), hclampy(iy+1), iz);       // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= (4 * aFourthOrder__ / (cx*cx*cy*cy)) * m_;


    ///////////////////////////////////////
    // Next-Next-Next Nearest Neighbours //
    ///////////////////////////////////////

    // Two over to left
    i_              = idx(lclampx(ix-2), iy, iz);                  // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= (2 * aFourthOrder__ / (cx*cx*cx*cx)) * m_;

    // Two over to right
    i_              = idx(hclampx(ix+2), iy, iz);                  // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= (2 * aFourthOrder__ / (cx*cx*cx*cx)) * m_;

    // Two below
    i_              = idx(ix, lclampy(iy-2), iz);                  // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= (2 * aFourthOrder__ / (cy*cy*cy*cy)) * m_;

    // Two above
    i_              = idx(ix, hclampy(iy+2), iz);                  // clamps or wraps index according to PBC
    m_              = make_float3(mx[i_], my[i_], mz[i_]);         // load m
    m_              = ( is0(m_)? m0: m_ );                         // replace missing non-boundary neighbor
    aFourthOrder__  = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B              -= (2 * aFourthOrder__ / (cy*cy*cy*cy)) * m_;


    float invMs = inv_Msat(Ms_, Ms_mul, I);

    Bx[I] += B.x*invMs;
    By[I] += B.y*invMs;
    Bz[I] += B.z*invMs;
}

