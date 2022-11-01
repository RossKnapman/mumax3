package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Pe = NewScalarParam("Pe", "C/m2", "Electric polarization")
	Ez = NewScalarParam("Ez", "V/m", "Electric field strength along the z-axis")
	B_elec = NewVectorField("B_elec", "T", "Effective magnetic field due to electric field", AddElectricEffectiveField)
	E_elec = NewScalarValue("E_elec", "J", "Electric field energy density", GetElectricFieldEnergy)
	Edens_elec = NewScalarField("Edens_elec", "J/m3", "Total electric field energy density", AddElectricFieldEnergyDensity)
)

func init () {
	registerEnergy(GetElectricFieldEnergy, AddElectricFieldEnergyDensity)
}

var AddElectricFieldEnergyDensity = makeEdensAdder(B_elec, -1)

func AddElectricEffectiveField(dst *data.Slice) {
	ms := Msat.MSlice()
	defer ms.Recycle()
	e := Ez.MSlice()
	defer e.Recycle()
	pe := Pe.MSlice()
	defer pe.Recycle()
	cuda.AddElectric(dst, M.Buffer(), e, pe, ms, regions.Gpu(), M.Mesh())
}

func GetElectricFieldEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_elec)
}
