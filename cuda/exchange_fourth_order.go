package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// Add NNN exchange field to Beff.
// see exchange_fourth_order.cu
func AddExchangeFourthOrder(B, m *data.Slice, I1_red SymmLUT, I2_red SymmLUT, Msat MSlice, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addexchangefourthorder_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		unsafe.Pointer(I1_red), unsafe.Pointer(I2_red), regions.Ptr,
		float32(c[X]), float32(c[Y]), float32(c[Z]), N[X], N[Y], N[Z], pbc, cfg)
}
