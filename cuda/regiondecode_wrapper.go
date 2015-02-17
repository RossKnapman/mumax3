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

// CUDA handle for regiondecode kernel
var regiondecode_code cu.Function

// Stores the arguments for regiondecode kernel invocation
type regiondecode_args_t struct {
	arg_dst     unsafe.Pointer
	arg_LUT     unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_N       int
	argptr      [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for regiondecode kernel invocation
var regiondecode_args regiondecode_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	regiondecode_args.argptr[0] = unsafe.Pointer(&regiondecode_args.arg_dst)
	regiondecode_args.argptr[1] = unsafe.Pointer(&regiondecode_args.arg_LUT)
	regiondecode_args.argptr[2] = unsafe.Pointer(&regiondecode_args.arg_regions)
	regiondecode_args.argptr[3] = unsafe.Pointer(&regiondecode_args.arg_N)
}

// Wrapper for regiondecode CUDA kernel, asynchronous.
func k_regiondecode_async(dst unsafe.Pointer, LUT unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("regiondecode")
	}

	regiondecode_args.Lock()
	defer regiondecode_args.Unlock()

	if regiondecode_code == 0 {
		regiondecode_code = fatbinLoad(regiondecode_map, "regiondecode")
	}

	regiondecode_args.arg_dst = dst
	regiondecode_args.arg_LUT = LUT
	regiondecode_args.arg_regions = regions
	regiondecode_args.arg_N = N

	args := regiondecode_args.argptr[:]
	cu.LaunchKernel(regiondecode_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("regiondecode")
	}
}

// maps compute capability on PTX code for regiondecode kernel.
var regiondecode_map = map[int]string{0: "",
	20: regiondecode_ptx_20,
	30: regiondecode_ptx_30,
	35: regiondecode_ptx_35,
	50: regiondecode_ptx_50}

// regiondecode PTX code for various compute capabilities.
const (
	regiondecode_ptx_20 = `
.version 4.1
.target sm_20
.address_size 64


.visible .entry regiondecode(
	.param .u64 regiondecode_param_0,
	.param .u64 regiondecode_param_1,
	.param .u64 regiondecode_param_2,
	.param .u32 regiondecode_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<2>;
	.reg .s64 	%rd<14>;


	ld.param.u64 	%rd1, [regiondecode_param_0];
	ld.param.u64 	%rd2, [regiondecode_param_1];
	ld.param.u64 	%rd3, [regiondecode_param_2];
	ld.param.u32 	%r2, [regiondecode_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd3;
	cvt.s64.s32	%rd7, %r1;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.u8 	%rd9, [%rd8];
	shl.b64 	%rd10, %rd9, 2;
	add.s64 	%rd11, %rd5, %rd10;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd4, %rd12;
	ld.global.f32 	%f1, [%rd11];
	st.global.f32 	[%rd13], %f1;

BB0_2:
	ret;
}


`
	regiondecode_ptx_30 = `
.version 4.1
.target sm_30
.address_size 64


.visible .entry regiondecode(
	.param .u64 regiondecode_param_0,
	.param .u64 regiondecode_param_1,
	.param .u64 regiondecode_param_2,
	.param .u32 regiondecode_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<2>;
	.reg .s64 	%rd<14>;


	ld.param.u64 	%rd1, [regiondecode_param_0];
	ld.param.u64 	%rd2, [regiondecode_param_1];
	ld.param.u64 	%rd3, [regiondecode_param_2];
	ld.param.u32 	%r2, [regiondecode_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd3;
	cvt.s64.s32	%rd7, %r1;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.u8 	%rd9, [%rd8];
	shl.b64 	%rd10, %rd9, 2;
	add.s64 	%rd11, %rd5, %rd10;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd4, %rd12;
	ld.global.f32 	%f1, [%rd11];
	st.global.f32 	[%rd13], %f1;

BB0_2:
	ret;
}


`
	regiondecode_ptx_35 = `
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

.visible .entry regiondecode(
	.param .u64 regiondecode_param_0,
	.param .u64 regiondecode_param_1,
	.param .u64 regiondecode_param_2,
	.param .u32 regiondecode_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .s16 	%rs<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<2>;
	.reg .s64 	%rd<15>;


	ld.param.u64 	%rd1, [regiondecode_param_0];
	ld.param.u64 	%rd2, [regiondecode_param_1];
	ld.param.u64 	%rd3, [regiondecode_param_2];
	ld.param.u32 	%r2, [regiondecode_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB5_2;

	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd3;
	cvt.s64.s32	%rd7, %r1;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.u8 	%rs1, [%rd8];
	cvt.u64.u16	%rd9, %rs1;
	and.b64  	%rd10, %rd9, 255;
	shl.b64 	%rd11, %rd10, 2;
	add.s64 	%rd12, %rd5, %rd11;
	mul.wide.s32 	%rd13, %r1, 4;
	add.s64 	%rd14, %rd4, %rd13;
	ld.global.nc.f32 	%f1, [%rd12];
	st.global.f32 	[%rd14], %f1;

BB5_2:
	ret;
}


`
	regiondecode_ptx_50 = `
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

.visible .entry regiondecode(
	.param .u64 regiondecode_param_0,
	.param .u64 regiondecode_param_1,
	.param .u64 regiondecode_param_2,
	.param .u32 regiondecode_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .s16 	%rs<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<2>;
	.reg .s64 	%rd<15>;


	ld.param.u64 	%rd1, [regiondecode_param_0];
	ld.param.u64 	%rd2, [regiondecode_param_1];
	ld.param.u64 	%rd3, [regiondecode_param_2];
	ld.param.u32 	%r2, [regiondecode_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB5_2;

	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd3;
	cvt.s64.s32	%rd7, %r1;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.u8 	%rs1, [%rd8];
	cvt.u64.u16	%rd9, %rs1;
	and.b64  	%rd10, %rd9, 255;
	shl.b64 	%rd11, %rd10, 2;
	add.s64 	%rd12, %rd5, %rd11;
	mul.wide.s32 	%rd13, %r1, 4;
	add.s64 	%rd14, %rd4, %rd13;
	ld.global.nc.f32 	%f1, [%rd12];
	st.global.f32 	[%rd14], %f1;

BB5_2:
	ret;
}


`
)
