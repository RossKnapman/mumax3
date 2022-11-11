package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Ered = NewScalarParam("Ered", "J/m2", "Electric field (V/m) * Polarization (C/m2) * Lattice constant (m)", &ered)
	ered exchParam

	B_elec     = NewVectorField("B_elec", "T", "Effective magnetic field due to electric field", AddElectricEffectiveField)
	E_elec     = NewScalarValue("E_elec", "J", "Electric field energy density", GetElectricFieldEnergy)
	Edens_elec = NewScalarField("Edens_elec", "J/m3", "Total electric field energy density", AddElectricFieldEnergyDensity)
)

func init() {
	registerEnergy(GetElectricFieldEnergy, AddElectricFieldEnergyDensity)
	ered.init(Ered)
}

var AddElectricFieldEnergyDensity = makeEdensAdder(B_elec, -0.5)

func AddElectricEffectiveField(dst *data.Slice) {
	ms := Msat.MSlice()
	defer ms.Recycle()
	cuda.AddElectric(dst, M.Buffer(), ered.Gpu(), ms, regions.Gpu(), M.Mesh())
}

func GetElectricFieldEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_elec)
}
