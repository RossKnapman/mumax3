#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange_fourth_order.go for more details.

extern "C" __global__ void
addexchangefourthorder(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ I1, float* __restrict__ I2,
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
    float I1__;  // second-order exchange stiffness
    float I2__;  // fourth-order exchange stiffness


    //////////////////
    // Central Spin //
    //////////////////
    i_    = idx(ix, iy, iz);
    m_    = make_float3(mx[i_], my[i_], mz[i_]);                                    // load m
    m_    = ( is0(m_)? m0: m_ );                                                    // replace missing non-boundary neighbor
    I1__  = I1[symidx(r0, regions[i_])];
    I2__  = I2[symidx(r0, regions[i_])];
    B    += 2 * I1__ * (1/(cx*cx) + 1/(cy*cy) + 1/(cz*cz)) * m_;
    B    -= 6 * I2__ * (1/(cx*cx*cx*cx) + 1/(cy*cy*cy*cy) + 1/(cz*cz*cz*cz)) * m_;
    B    -= 8 * I2__ * (1/(cx*cx*cy*cy) + 1/(cx*cx*cz*cz) + 1/(cy*cy*cz*cz)) * m_;


    ///////////////////////////////
    // Direct Nearest Neighbours //
    ///////////////////////////////
    
    // Left neighbour
    i_    = idx(lclampx(ix-1), iy, iz);                                             // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);                                    // load m
    m_    = ( is0(m_)? m0: m_ );                                                    // replace missing non-boundary neighbor
    I1__  = I1[symidx(r0, regions[i_])];
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I1__ / (cx*cx)) * m_;
    B    += 4 * I2__ * (1/(cx*cx*cx*cx) + 1/(cx*cx*cy*cy) + 1/(cx*cx*cz*cz)) * m_;

    // Right neighbour
    i_    = idx(hclampx(ix+1), iy, iz);                                             // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);                                    // load m
    m_    = ( is0(m_)? m0: m_ );                                                    // replace missing non-boundary neighbor
    I1__  = I1[symidx(r0, regions[i_])];
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I1__ / (cx*cx)) * m_;
    B    += 4 * I2__ * (1/(cx*cx*cx*cx) + 1/(cx*cx*cy*cy) + 1/(cx*cx*cz*cz)) * m_;

    // Below neighbour
    i_    = idx(ix, lclampy(iy-1), iz);                                             // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);                                    // load m
    m_    = ( is0(m_)? m0: m_ );                                                    // replace missing non-boundary neighbor
    I1__  = I1[symidx(r0, regions[i_])];
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I1__ / (cy*cy)) * m_;
    B    += 4 * I2__ * (1/(cy*cy*cy*cy) + 1/(cy*cy*cx*cx) + 1/(cy*cy*cz*cz)) * m_;

    // Above neighbour
    i_    = idx(ix, hclampy(iy+1), iz);                                             // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);                                    // load m
    m_    = ( is0(m_)? m0: m_ );                                                    // replace missing non-boundary neighbor
    I1__  = I1[symidx(r0, regions[i_])];
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I1__ / (cy*cy)) * m_;
    B    += 4 * I2__ * (1/(cy*cy*cy*cy) + 1/(cy*cy*cx*cx) + 1/(cy*cy*cz*cz)) * m_;

    // Bottom neighbour
    i_    = idx(ix, iy, lclampz(iz-1));                                             // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);                                    // load m
    m_    = ( is0(m_)? m0: m_ );                                                    // replace missing non-boundary neighbor
    I1__  = I1[symidx(r0, regions[i_])];
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I1__ / (cz*cz)) * m_;
    B    += 4 * I2__ * (1/(cz*cz*cz*cz) + 1/(cz*cz*cx*cx) + 1/(cz*cz*cy*cy)) * m_;

    // Top neighbour
    i_    = idx(ix, iy, hclampz(iz+1));                                             // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);                                    // load m
    m_    = ( is0(m_)? m0: m_ );                                                    // replace missing non-boundary neighbor
    I1__  = I1[symidx(r0, regions[i_])];
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I1__ / (cz*cz)) * m_;
    B    += 4 * I2__ * (1/(cz*cz*cz*cz) + 1/(cz*cz*cx*cx) + 1/(cz*cz*cy*cy)) * m_;


    //////////////////////////////////////////////
    // Diagonal Nearest Neighbours in z=0 Plane //
    //////////////////////////////////////////////

    // Bottom-left neighbour
    i_    = idx(lclampx(ix-1), lclampy(iy-1), iz);  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cx*cx*cy*cy)) * m_;

    // Top-left neighbour
    i_    = idx(lclampx(ix-1), hclampy(iy+1), iz);  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cx*cx*cy*cy)) * m_;

    // Bottom-right neighbour
    i_    = idx(hclampx(ix+1), lclampy(iy-1), iz);  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cx*cx*cy*cy)) * m_;

    // Top-right neighbour
    i_    = idx(hclampx(ix+1), hclampy(iy+1), iz);  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cx*cx*cy*cy)) * m_;


    //////////////////////////////////////////////
    // Diagonal Nearest Neighbours in x=0 Plane //
    //////////////////////////////////////////////

    // Bottom-left neighbour
    i_    = idx(ix, lclampy(iy-1), hclampz(iz+1));  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cy*cy*cz*cz)) * m_;

    // Top-left neighbour
    i_    = idx(ix, hclampy(iy+1), hclampz(iz+1));  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cy*cy*cz*cz)) * m_;

    // Bottom-right neighbour
    i_    = idx(ix, lclampy(iy-1), lclampz(iz-1));  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cy*cy*cz*cz)) * m_;

    // Top-right neighbour
    i_    = idx(ix, hclampy(iy+1), lclampz(iz-1));  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cy*cy*cz*cz)) * m_;


    //////////////////////////////////////////////
    // Diagonal Nearest Neighbours in y=0 Plane //
    //////////////////////////////////////////////

    // Bottom-left neighbour
    i_    = idx(lclampx(ix-1), iy, hclampz(iz+1));  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cx*cx*cz*cz)) * m_;

    // Top-left neighbour
    i_    = idx(lclampx(ix-1), iy, lclampz(iz-1));  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cx*cx*cz*cz)) * m_;

    // Bottom-right neighbour
    i_    = idx(hclampx(ix+1), iy, hclampz(iz+1));  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cx*cx*cz*cz)) * m_;

    // Top-right neighbour
    i_    = idx(hclampx(ix+1), iy, lclampz(iz-1));  // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);    // load m
    m_    = ( is0(m_)? m0: m_ );                    // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (2 * I2__ / (cx*cx*cz*cz)) * m_;


    ///////////////////////////////////////
    // Next-Next-Next Nearest Neighbours //
    ///////////////////////////////////////

    // Two over to left
    i_    = idx(lclampx(ix-2), iy, iz);           // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_    = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I2__ / (cx*cx*cx*cx)) * m_;

    // Two over to right
    i_    = idx(hclampx(ix+2), iy, iz);           // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_    = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I2__ / (cx*cx*cx*cx)) * m_;

    // Two below
    i_    = idx(ix, lclampy(iy-2), iz);           // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_    = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I2__ / (cy*cy*cy*cy)) * m_;

    // Two above
    i_    = idx(ix, hclampy(iy+2), iz);           // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_    = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I2__ / (cy*cy*cy*cy)) * m_;

    // Two bottom
    i_    = idx(ix, iy, lclampz(iz-2));           // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_    = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I2__ / (cz*cz*cz*cz)) * m_;

    // Two top
    i_    = idx(ix, iy, hclampz(iz+2));           // clamps or wraps index according to PBC
    m_    = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_    = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    I2__  = I2[symidx(r0, regions[i_])];
    B    -= (I2__ / (cz*cz*cz*cz)) * m_;


    float invMs = inv_Msat(Ms_, Ms_mul, I);

    Bx[I] += B.x*invMs;
    By[I] += B.y*invMs;
    Bz[I] += B.z*invMs;

}

