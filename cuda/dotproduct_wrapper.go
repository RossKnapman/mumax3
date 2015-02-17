package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for dotproduct kernel
var dotproduct_code cu.Function

// Stores the arguments for dotproduct kernel invocation
type dotproduct_args_t struct {
	arg_dst       unsafe.Pointer
	arg_prefactor float32
	arg_ax        unsafe.Pointer
	arg_ay        unsafe.Pointer
	arg_az        unsafe.Pointer
	arg_bx        unsafe.Pointer
	arg_by        unsafe.Pointer
	arg_bz        unsafe.Pointer
	arg_N         int
	argptr        [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for dotproduct kernel invocation
var dotproduct_args dotproduct_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	dotproduct_args.argptr[0] = unsafe.Pointer(&dotproduct_args.arg_dst)
	dotproduct_args.argptr[1] = unsafe.Pointer(&dotproduct_args.arg_prefactor)
	dotproduct_args.argptr[2] = unsafe.Pointer(&dotproduct_args.arg_ax)
	dotproduct_args.argptr[3] = unsafe.Pointer(&dotproduct_args.arg_ay)
	dotproduct_args.argptr[4] = unsafe.Pointer(&dotproduct_args.arg_az)
	dotproduct_args.argptr[5] = unsafe.Pointer(&dotproduct_args.arg_bx)
	dotproduct_args.argptr[6] = unsafe.Pointer(&dotproduct_args.arg_by)
	dotproduct_args.argptr[7] = unsafe.Pointer(&dotproduct_args.arg_bz)
	dotproduct_args.argptr[8] = unsafe.Pointer(&dotproduct_args.arg_N)
}

// Wrapper for dotproduct CUDA kernel, asynchronous.
func k_dotproduct_async(dst unsafe.Pointer, prefactor float32, ax unsafe.Pointer, ay unsafe.Pointer, az unsafe.Pointer, bx unsafe.Pointer, by unsafe.Pointer, bz unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("dotproduct")
	}

	dotproduct_args.Lock()
	defer dotproduct_args.Unlock()

	if dotproduct_code == 0 {
		dotproduct_code = fatbinLoad(dotproduct_map, "dotproduct")
	}

	dotproduct_args.arg_dst = dst
	dotproduct_args.arg_prefactor = prefactor
	dotproduct_args.arg_ax = ax
	dotproduct_args.arg_ay = ay
	dotproduct_args.arg_az = az
	dotproduct_args.arg_bx = bx
	dotproduct_args.arg_by = by
	dotproduct_args.arg_bz = bz
	dotproduct_args.arg_N = N

	args := dotproduct_args.argptr[:]
	cu.LaunchKernel(dotproduct_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("dotproduct")
	}
}

// maps compute capability on PTX code for dotproduct kernel.
var dotproduct_map = map[int]string{0: "",
	20: dotproduct_ptx_20,
	30: dotproduct_ptx_30,
	35: dotproduct_ptx_35,
	50: dotproduct_ptx_50}

// dotproduct PTX code for various compute capabilities.
const (
	dotproduct_ptx_20 = `
.version 4.1
.target sm_20
.address_size 64


.visible .entry dotproduct(
	.param .u64 dotproduct_param_0,
	.param .f32 dotproduct_param_1,
	.param .u64 dotproduct_param_2,
	.param .u64 dotproduct_param_3,
	.param .u64 dotproduct_param_4,
	.param .u64 dotproduct_param_5,
	.param .u64 dotproduct_param_6,
	.param .u64 dotproduct_param_7,
	.param .u32 dotproduct_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<13>;
	.reg .s64 	%rd<23>;


	ld.param.u64 	%rd1, [dotproduct_param_0];
	ld.param.f32 	%f1, [dotproduct_param_1];
	ld.param.u64 	%rd2, [dotproduct_param_2];
	ld.param.u64 	%rd3, [dotproduct_param_3];
	ld.param.u64 	%rd4, [dotproduct_param_4];
	ld.param.u64 	%rd5, [dotproduct_param_5];
	ld.param.u64 	%rd6, [dotproduct_param_6];
	ld.param.u64 	%rd7, [dotproduct_param_7];
	ld.param.u32 	%r2, [dotproduct_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd1;
	cvta.to.global.u64 	%rd9, %rd7;
	cvta.to.global.u64 	%rd10, %rd6;
	cvta.to.global.u64 	%rd11, %rd5;
	cvta.to.global.u64 	%rd12, %rd4;
	cvta.to.global.u64 	%rd13, %rd3;
	cvta.to.global.u64 	%rd14, %rd2;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	add.s64 	%rd17, %rd13, %rd15;
	add.s64 	%rd18, %rd12, %rd15;
	add.s64 	%rd19, %rd11, %rd15;
	add.s64 	%rd20, %rd10, %rd15;
	add.s64 	%rd21, %rd9, %rd15;
	ld.global.f32 	%f2, [%rd19];
	ld.global.f32 	%f3, [%rd16];
	ld.global.f32 	%f4, [%rd20];
	ld.global.f32 	%f5, [%rd17];
	mul.f32 	%f6, %f5, %f4;
	fma.rn.f32 	%f7, %f3, %f2, %f6;
	ld.global.f32 	%f8, [%rd21];
	ld.global.f32 	%f9, [%rd18];
	fma.rn.f32 	%f10, %f9, %f8, %f7;
	add.s64 	%rd22, %rd8, %rd15;
	ld.global.f32 	%f11, [%rd22];
	fma.rn.f32 	%f12, %f10, %f1, %f11;
	st.global.f32 	[%rd22], %f12;

BB0_2:
	ret;
}


`
	dotproduct_ptx_30 = `
.version 4.1
.target sm_30
.address_size 64


.visible .entry dotproduct(
	.param .u64 dotproduct_param_0,
	.param .f32 dotproduct_param_1,
	.param .u64 dotproduct_param_2,
	.param .u64 dotproduct_param_3,
	.param .u64 dotproduct_param_4,
	.param .u64 dotproduct_param_5,
	.param .u64 dotproduct_param_6,
	.param .u64 dotproduct_param_7,
	.param .u32 dotproduct_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<13>;
	.reg .s64 	%rd<23>;


	ld.param.u64 	%rd1, [dotproduct_param_0];
	ld.param.f32 	%f1, [dotproduct_param_1];
	ld.param.u64 	%rd2, [dotproduct_param_2];
	ld.param.u64 	%rd3, [dotproduct_param_3];
	ld.param.u64 	%rd4, [dotproduct_param_4];
	ld.param.u64 	%rd5, [dotproduct_param_5];
	ld.param.u64 	%rd6, [dotproduct_param_6];
	ld.param.u64 	%rd7, [dotproduct_param_7];
	ld.param.u32 	%r2, [dotproduct_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd1;
	cvta.to.global.u64 	%rd9, %rd7;
	cvta.to.global.u64 	%rd10, %rd6;
	cvta.to.global.u64 	%rd11, %rd5;
	cvta.to.global.u64 	%rd12, %rd4;
	cvta.to.global.u64 	%rd13, %rd3;
	cvta.to.global.u64 	%rd14, %rd2;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	add.s64 	%rd17, %rd13, %rd15;
	add.s64 	%rd18, %rd12, %rd15;
	add.s64 	%rd19, %rd11, %rd15;
	add.s64 	%rd20, %rd10, %rd15;
	add.s64 	%rd21, %rd9, %rd15;
	ld.global.f32 	%f2, [%rd19];
	ld.global.f32 	%f3, [%rd16];
	ld.global.f32 	%f4, [%rd20];
	ld.global.f32 	%f5, [%rd17];
	mul.f32 	%f6, %f5, %f4;
	fma.rn.f32 	%f7, %f3, %f2, %f6;
	ld.global.f32 	%f8, [%rd21];
	ld.global.f32 	%f9, [%rd18];
	fma.rn.f32 	%f10, %f9, %f8, %f7;
	add.s64 	%rd22, %rd8, %rd15;
	ld.global.f32 	%f11, [%rd22];
	fma.rn.f32 	%f12, %f10, %f1, %f11;
	st.global.f32 	[%rd22], %f12;

BB0_2:
	ret;
}


`
	dotproduct_ptx_35 = `
.version 4.1
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.visible .entry dotproduct(
	.param .u64 dotproduct_param_0,
	.param .f32 dotproduct_param_1,
	.param .u64 dotproduct_param_2,
	.param .u64 dotproduct_param_3,
	.param .u64 dotproduct_param_4,
	.param .u64 dotproduct_param_5,
	.param .u64 dotproduct_param_6,
	.param .u64 dotproduct_param_7,
	.param .u32 dotproduct_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<13>;
	.reg .s64 	%rd<23>;


	ld.param.u64 	%rd1, [dotproduct_param_0];
	ld.param.f32 	%f1, [dotproduct_param_1];
	ld.param.u64 	%rd2, [dotproduct_param_2];
	ld.param.u64 	%rd3, [dotproduct_param_3];
	ld.param.u64 	%rd4, [dotproduct_param_4];
	ld.param.u64 	%rd5, [dotproduct_param_5];
	ld.param.u64 	%rd6, [dotproduct_param_6];
	ld.param.u64 	%rd7, [dotproduct_param_7];
	ld.param.u32 	%r2, [dotproduct_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB5_2;

	cvta.to.global.u64 	%rd8, %rd1;
	cvta.to.global.u64 	%rd9, %rd7;
	cvta.to.global.u64 	%rd10, %rd6;
	cvta.to.global.u64 	%rd11, %rd5;
	cvta.to.global.u64 	%rd12, %rd4;
	cvta.to.global.u64 	%rd13, %rd3;
	cvta.to.global.u64 	%rd14, %rd2;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	add.s64 	%rd17, %rd13, %rd15;
	add.s64 	%rd18, %rd12, %rd15;
	add.s64 	%rd19, %rd11, %rd15;
	add.s64 	%rd20, %rd10, %rd15;
	add.s64 	%rd21, %rd9, %rd15;
	ld.global.nc.f32 	%f2, [%rd19];
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd17];
	mul.f32 	%f6, %f5, %f4;
	fma.rn.f32 	%f7, %f3, %f2, %f6;
	ld.global.nc.f32 	%f8, [%rd21];
	ld.global.nc.f32 	%f9, [%rd18];
	fma.rn.f32 	%f10, %f9, %f8, %f7;
	add.s64 	%rd22, %rd8, %rd15;
	ld.global.f32 	%f11, [%rd22];
	fma.rn.f32 	%f12, %f10, %f1, %f11;
	st.global.f32 	[%rd22], %f12;

BB5_2:
	ret;
}


`
	dotproduct_ptx_50 = `
.version 4.1
.target sm_50
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.visible .entry dotproduct(
	.param .u64 dotproduct_param_0,
	.param .f32 dotproduct_param_1,
	.param .u64 dotproduct_param_2,
	.param .u64 dotproduct_param_3,
	.param .u64 dotproduct_param_4,
	.param .u64 dotproduct_param_5,
	.param .u64 dotproduct_param_6,
	.param .u64 dotproduct_param_7,
	.param .u32 dotproduct_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<13>;
	.reg .s64 	%rd<23>;


	ld.param.u64 	%rd1, [dotproduct_param_0];
	ld.param.f32 	%f1, [dotproduct_param_1];
	ld.param.u64 	%rd2, [dotproduct_param_2];
	ld.param.u64 	%rd3, [dotproduct_param_3];
	ld.param.u64 	%rd4, [dotproduct_param_4];
	ld.param.u64 	%rd5, [dotproduct_param_5];
	ld.param.u64 	%rd6, [dotproduct_param_6];
	ld.param.u64 	%rd7, [dotproduct_param_7];
	ld.param.u32 	%r2, [dotproduct_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB5_2;

	cvta.to.global.u64 	%rd8, %rd1;
	cvta.to.global.u64 	%rd9, %rd7;
	cvta.to.global.u64 	%rd10, %rd6;
	cvta.to.global.u64 	%rd11, %rd5;
	cvta.to.global.u64 	%rd12, %rd4;
	cvta.to.global.u64 	%rd13, %rd3;
	cvta.to.global.u64 	%rd14, %rd2;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	add.s64 	%rd17, %rd13, %rd15;
	add.s64 	%rd18, %rd12, %rd15;
	add.s64 	%rd19, %rd11, %rd15;
	add.s64 	%rd20, %rd10, %rd15;
	add.s64 	%rd21, %rd9, %rd15;
	ld.global.nc.f32 	%f2, [%rd19];
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd17];
	mul.f32 	%f6, %f5, %f4;
	fma.rn.f32 	%f7, %f3, %f2, %f6;
	ld.global.nc.f32 	%f8, [%rd21];
	ld.global.nc.f32 	%f9, [%rd18];
	fma.rn.f32 	%f10, %f9, %f8, %f7;
	add.s64 	%rd22, %rd8, %rd15;
	ld.global.f32 	%f11, [%rd22];
	fma.rn.f32 	%f12, %f10, %f1, %f11;
	st.global.f32 	[%rd22], %f12;

BB5_2:
	ret;
}


`
)
