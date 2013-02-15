package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"math"
	"unsafe"
)

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp int, m *data.Mesh) *data.Slice {
	return newSlice(nComp, m, memAlloc, data.GPUMemory)
}

// Make a GPU Slice with nComp components each of size length.
func NewUnifiedSlice(nComp int, m *data.Mesh) *data.Slice {
	return newSlice(nComp, m, cu.MemAllocHost, data.UnifiedMemory)
}

func newSlice(nComp int, m *data.Mesh, alloc func(int64) unsafe.Pointer, memType int8) *data.Slice {
	data.EnableGPU(memFree, cu.MemFreeHost)
	length := m.NCell()
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	for c := range ptrs {
		ptrs[c] = unsafe.Pointer(alloc(bytes))
		cu.MemsetD32(cu.DevicePtr(ptrs[c]), 0, int64(length))
	}
	return data.SliceFromPtrs(m, memType, ptrs)
}

func memFree(ptr unsafe.Pointer) { cu.MemFree(cu.DevicePtr(ptr)) }

// Wrapper for cu.MemAlloc, fatal exit on out of memory.
func memAlloc(bytes int64) unsafe.Pointer {
	defer func() {
		err := recover()
		if err == cu.ERROR_OUT_OF_MEMORY {
			log.Fatal(err)
		}
		if err != nil {
			panic(err)
		}
	}()
	return unsafe.Pointer(cu.MemAlloc(bytes))
}

// Memset sets the Slice's components to the specified values.
func Memset(s *data.Slice, val ...float32) {
	util.Argument(len(val) == s.NComp())
	str := kernel.Stream()
	for c, v := range val {
		cu.MemsetD32Async(cu.DevicePtr(s.DevPtr(c)), math.Float32bits(v), int64(s.Len()), str)
	}
	kernel.SyncAndRecycle(str)
}
