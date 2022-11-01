package cuda

import (

	"github.com/mumax/3/data"
)

// Add electric term to Beff.
// see electric.cu
func AddElectric(B, m *data.Slice, E, Pe, Msat MSlice, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addelectric_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		E.DevPtr(0), Pe.DevPtr(0), regions.Ptr,
		float32(c[X]), float32(c[Y]), float32(c[Z]), N[X], N[Y], N[Z], pbc, cfg)
}
