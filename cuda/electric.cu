#include <stdint.h>
#include "exchange.h"
#include "stencil.h"
#include "amul.h"

// Electric field term according to
// Katsura, Nagaosa, Balatsky, Phys. Rev. Lett. 95, 057205 (2005).

extern "C" __global__ void
addelectric(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ eLUT2d, uint8_t* __restrict__ regions,
            float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    uint8_t r0 = regions[I];
    float3 B  = make_float3(0.0,0.0,0.0);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float ered__; // reduced electric field


    ///////////////////
    // x derivatives //
    ///////////////////

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    ered__ = eLUT2d[symidx(r0, regions[i_])];
    B.x -= (ered__/cx) * m_.z;
    B.z += (ered__/cx) * m_.x;

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    ered__ = eLUT2d[symidx(r0, regions[i_])];
    B.x += (ered__/cx) * m_.z;
    B.z -= (ered__/cx) * m_.x;
    
    ///////////////////
    // y derivatives //
    ///////////////////

    // above neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    ered__ = eLUT2d[symidx(r0, regions[i_])];
    B.y -= (ered__/cy) * m_.z;
    B.z += (ered__/cy) * m_.y;

    // below neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    ered__ = eLUT2d[symidx(r0, regions[i_])];
    B.y += (ered__/cy) * m_.z;
    B.z -= (ered__/cy) * m_.y;


    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Bx[I] += B.x*invMs;
    By[I] += B.y*invMs;
    Bz[I] += B.z*invMs;
}
