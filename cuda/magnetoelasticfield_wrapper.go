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

// CUDA handle for addmagnetoelasticfield kernel
var addmagnetoelasticfield_code cu.Function

// Stores the arguments for addmagnetoelasticfield kernel invocation
type addmagnetoelasticfield_args_t struct {
	arg_Bx      unsafe.Pointer
	arg_By      unsafe.Pointer
	arg_Bz      unsafe.Pointer
	arg_mx      unsafe.Pointer
	arg_my      unsafe.Pointer
	arg_mz      unsafe.Pointer
	arg_exx_    unsafe.Pointer
	arg_exx_mul float32
	arg_eyy_    unsafe.Pointer
	arg_eyy_mul float32
	arg_ezz_    unsafe.Pointer
	arg_ezz_mul float32
	arg_exy_    unsafe.Pointer
	arg_exy_mul float32
	arg_exz_    unsafe.Pointer
	arg_exz_mul float32
	arg_eyz_    unsafe.Pointer
	arg_eyz_mul float32
	arg_B1_     unsafe.Pointer
	arg_B1_mul  float32
	arg_B2_     unsafe.Pointer
	arg_B2_mul  float32
	arg_Ms_     unsafe.Pointer
	arg_Ms_mul  float32
	arg_N       int
	argptr      [25]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addmagnetoelasticfield kernel invocation
var addmagnetoelasticfield_args addmagnetoelasticfield_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	addmagnetoelasticfield_args.argptr[0] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_Bx)
	addmagnetoelasticfield_args.argptr[1] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_By)
	addmagnetoelasticfield_args.argptr[2] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_Bz)
	addmagnetoelasticfield_args.argptr[3] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_mx)
	addmagnetoelasticfield_args.argptr[4] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_my)
	addmagnetoelasticfield_args.argptr[5] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_mz)
	addmagnetoelasticfield_args.argptr[6] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exx_)
	addmagnetoelasticfield_args.argptr[7] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exx_mul)
	addmagnetoelasticfield_args.argptr[8] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_eyy_)
	addmagnetoelasticfield_args.argptr[9] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_eyy_mul)
	addmagnetoelasticfield_args.argptr[10] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_ezz_)
	addmagnetoelasticfield_args.argptr[11] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_ezz_mul)
	addmagnetoelasticfield_args.argptr[12] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exy_)
	addmagnetoelasticfield_args.argptr[13] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exy_mul)
	addmagnetoelasticfield_args.argptr[14] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exz_)
	addmagnetoelasticfield_args.argptr[15] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_exz_mul)
	addmagnetoelasticfield_args.argptr[16] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_eyz_)
	addmagnetoelasticfield_args.argptr[17] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_eyz_mul)
	addmagnetoelasticfield_args.argptr[18] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_B1_)
	addmagnetoelasticfield_args.argptr[19] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_B1_mul)
	addmagnetoelasticfield_args.argptr[20] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_B2_)
	addmagnetoelasticfield_args.argptr[21] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_B2_mul)
	addmagnetoelasticfield_args.argptr[22] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_Ms_)
	addmagnetoelasticfield_args.argptr[23] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_Ms_mul)
	addmagnetoelasticfield_args.argptr[24] = unsafe.Pointer(&addmagnetoelasticfield_args.arg_N)
}

// Wrapper for addmagnetoelasticfield CUDA kernel, asynchronous.
func k_addmagnetoelasticfield_async(Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, exx_ unsafe.Pointer, exx_mul float32, eyy_ unsafe.Pointer, eyy_mul float32, ezz_ unsafe.Pointer, ezz_mul float32, exy_ unsafe.Pointer, exy_mul float32, exz_ unsafe.Pointer, exz_mul float32, eyz_ unsafe.Pointer, eyz_mul float32, B1_ unsafe.Pointer, B1_mul float32, B2_ unsafe.Pointer, B2_mul float32, Ms_ unsafe.Pointer, Ms_mul float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("addmagnetoelasticfield")
	}

	addmagnetoelasticfield_args.Lock()
	defer addmagnetoelasticfield_args.Unlock()

	if addmagnetoelasticfield_code == 0 {
		addmagnetoelasticfield_code = fatbinLoad(addmagnetoelasticfield_map, "addmagnetoelasticfield")
	}

	addmagnetoelasticfield_args.arg_Bx = Bx
	addmagnetoelasticfield_args.arg_By = By
	addmagnetoelasticfield_args.arg_Bz = Bz
	addmagnetoelasticfield_args.arg_mx = mx
	addmagnetoelasticfield_args.arg_my = my
	addmagnetoelasticfield_args.arg_mz = mz
	addmagnetoelasticfield_args.arg_exx_ = exx_
	addmagnetoelasticfield_args.arg_exx_mul = exx_mul
	addmagnetoelasticfield_args.arg_eyy_ = eyy_
	addmagnetoelasticfield_args.arg_eyy_mul = eyy_mul
	addmagnetoelasticfield_args.arg_ezz_ = ezz_
	addmagnetoelasticfield_args.arg_ezz_mul = ezz_mul
	addmagnetoelasticfield_args.arg_exy_ = exy_
	addmagnetoelasticfield_args.arg_exy_mul = exy_mul
	addmagnetoelasticfield_args.arg_exz_ = exz_
	addmagnetoelasticfield_args.arg_exz_mul = exz_mul
	addmagnetoelasticfield_args.arg_eyz_ = eyz_
	addmagnetoelasticfield_args.arg_eyz_mul = eyz_mul
	addmagnetoelasticfield_args.arg_B1_ = B1_
	addmagnetoelasticfield_args.arg_B1_mul = B1_mul
	addmagnetoelasticfield_args.arg_B2_ = B2_
	addmagnetoelasticfield_args.arg_B2_mul = B2_mul
	addmagnetoelasticfield_args.arg_Ms_ = Ms_
	addmagnetoelasticfield_args.arg_Ms_mul = Ms_mul
	addmagnetoelasticfield_args.arg_N = N

	args := addmagnetoelasticfield_args.argptr[:]
	cu.LaunchKernel(addmagnetoelasticfield_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("addmagnetoelasticfield")
	}
}

// maps compute capability on PTX code for addmagnetoelasticfield kernel.
var addmagnetoelasticfield_map = map[int]string{0: "",
	35: addmagnetoelasticfield_ptx_35}

// addmagnetoelasticfield PTX code for various compute capabilities.
const (
	addmagnetoelasticfield_ptx_35 = `
.version 7.5
.target sm_35
.address_size 64

	// .globl	addmagnetoelasticfield

.visible .entry addmagnetoelasticfield(
	.param .u64 addmagnetoelasticfield_param_0,
	.param .u64 addmagnetoelasticfield_param_1,
	.param .u64 addmagnetoelasticfield_param_2,
	.param .u64 addmagnetoelasticfield_param_3,
	.param .u64 addmagnetoelasticfield_param_4,
	.param .u64 addmagnetoelasticfield_param_5,
	.param .u64 addmagnetoelasticfield_param_6,
	.param .f32 addmagnetoelasticfield_param_7,
	.param .u64 addmagnetoelasticfield_param_8,
	.param .f32 addmagnetoelasticfield_param_9,
	.param .u64 addmagnetoelasticfield_param_10,
	.param .f32 addmagnetoelasticfield_param_11,
	.param .u64 addmagnetoelasticfield_param_12,
	.param .f32 addmagnetoelasticfield_param_13,
	.param .u64 addmagnetoelasticfield_param_14,
	.param .f32 addmagnetoelasticfield_param_15,
	.param .u64 addmagnetoelasticfield_param_16,
	.param .f32 addmagnetoelasticfield_param_17,
	.param .u64 addmagnetoelasticfield_param_18,
	.param .f32 addmagnetoelasticfield_param_19,
	.param .u64 addmagnetoelasticfield_param_20,
	.param .f32 addmagnetoelasticfield_param_21,
	.param .u64 addmagnetoelasticfield_param_22,
	.param .f32 addmagnetoelasticfield_param_23,
	.param .u32 addmagnetoelasticfield_param_24
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<77>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<56>;


	ld.param.u64 	%rd1, [addmagnetoelasticfield_param_0];
	ld.param.u64 	%rd2, [addmagnetoelasticfield_param_1];
	ld.param.u64 	%rd3, [addmagnetoelasticfield_param_2];
	ld.param.u64 	%rd4, [addmagnetoelasticfield_param_3];
	ld.param.u64 	%rd5, [addmagnetoelasticfield_param_4];
	ld.param.u64 	%rd6, [addmagnetoelasticfield_param_5];
	ld.param.u64 	%rd7, [addmagnetoelasticfield_param_6];
	ld.param.f32 	%f67, [addmagnetoelasticfield_param_7];
	ld.param.u64 	%rd8, [addmagnetoelasticfield_param_8];
	ld.param.f32 	%f68, [addmagnetoelasticfield_param_9];
	ld.param.u64 	%rd9, [addmagnetoelasticfield_param_10];
	ld.param.f32 	%f69, [addmagnetoelasticfield_param_11];
	ld.param.u64 	%rd10, [addmagnetoelasticfield_param_12];
	ld.param.f32 	%f70, [addmagnetoelasticfield_param_13];
	ld.param.u64 	%rd11, [addmagnetoelasticfield_param_14];
	ld.param.f32 	%f71, [addmagnetoelasticfield_param_15];
	ld.param.u64 	%rd12, [addmagnetoelasticfield_param_16];
	ld.param.f32 	%f72, [addmagnetoelasticfield_param_17];
	ld.param.u64 	%rd13, [addmagnetoelasticfield_param_18];
	ld.param.f32 	%f75, [addmagnetoelasticfield_param_19];
	ld.param.u64 	%rd14, [addmagnetoelasticfield_param_20];
	ld.param.f32 	%f76, [addmagnetoelasticfield_param_21];
	ld.param.u64 	%rd15, [addmagnetoelasticfield_param_22];
	ld.param.f32 	%f73, [addmagnetoelasticfield_param_23];
	ld.param.u32 	%r2, [addmagnetoelasticfield_param_24];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_22;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd16, %rd7;
	mul.wide.s32 	%rd17, %r1, 4;
	add.s64 	%rd18, %rd16, %rd17;
	ld.global.nc.f32 	%f30, [%rd18];
	mul.f32 	%f67, %f30, %f67;

$L__BB0_3:
	setp.eq.s64 	%p3, %rd8, 0;
	@%p3 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd19, %rd8;
	mul.wide.s32 	%rd20, %r1, 4;
	add.s64 	%rd21, %rd19, %rd20;
	ld.global.nc.f32 	%f31, [%rd21];
	mul.f32 	%f68, %f31, %f68;

$L__BB0_5:
	setp.eq.s64 	%p4, %rd9, 0;
	@%p4 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd22, %rd9;
	mul.wide.s32 	%rd23, %r1, 4;
	add.s64 	%rd24, %rd22, %rd23;
	ld.global.nc.f32 	%f32, [%rd24];
	mul.f32 	%f69, %f32, %f69;

$L__BB0_7:
	setp.eq.s64 	%p5, %rd10, 0;
	@%p5 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd25, %rd10;
	mul.wide.s32 	%rd26, %r1, 4;
	add.s64 	%rd27, %rd25, %rd26;
	ld.global.nc.f32 	%f33, [%rd27];
	mul.f32 	%f70, %f33, %f70;

$L__BB0_9:
	setp.eq.s64 	%p6, %rd11, 0;
	@%p6 bra 	$L__BB0_11;

	cvta.to.global.u64 	%rd28, %rd11;
	mul.wide.s32 	%rd29, %r1, 4;
	add.s64 	%rd30, %rd28, %rd29;
	ld.global.nc.f32 	%f34, [%rd30];
	mul.f32 	%f71, %f34, %f71;

$L__BB0_11:
	setp.eq.s64 	%p7, %rd12, 0;
	@%p7 bra 	$L__BB0_13;

	cvta.to.global.u64 	%rd31, %rd12;
	mul.wide.s32 	%rd32, %r1, 4;
	add.s64 	%rd33, %rd31, %rd32;
	ld.global.nc.f32 	%f35, [%rd33];
	mul.f32 	%f72, %f35, %f72;

$L__BB0_13:
	setp.eq.s64 	%p8, %rd15, 0;
	@%p8 bra 	$L__BB0_15;

	cvta.to.global.u64 	%rd34, %rd15;
	mul.wide.s32 	%rd35, %r1, 4;
	add.s64 	%rd36, %rd34, %rd35;
	ld.global.nc.f32 	%f36, [%rd36];
	mul.f32 	%f73, %f36, %f73;

$L__BB0_15:
	setp.eq.f32 	%p9, %f73, 0f00000000;
	mov.f32 	%f74, 0f00000000;
	@%p9 bra 	$L__BB0_17;

	rcp.rn.f32 	%f74, %f73;

$L__BB0_17:
	setp.eq.s64 	%p10, %rd13, 0;
	@%p10 bra 	$L__BB0_19;

	cvta.to.global.u64 	%rd37, %rd13;
	mul.wide.s32 	%rd38, %r1, 4;
	add.s64 	%rd39, %rd37, %rd38;
	ld.global.nc.f32 	%f38, [%rd39];
	mul.f32 	%f75, %f38, %f75;

$L__BB0_19:
	setp.eq.s64 	%p11, %rd14, 0;
	@%p11 bra 	$L__BB0_21;

	cvta.to.global.u64 	%rd40, %rd14;
	mul.wide.s32 	%rd41, %r1, 4;
	add.s64 	%rd42, %rd40, %rd41;
	ld.global.nc.f32 	%f39, [%rd42];
	mul.f32 	%f76, %f39, %f76;

$L__BB0_21:
	mul.f32 	%f40, %f74, %f75;
	fma.rn.f32 	%f41, %f74, %f75, %f40;
	cvta.to.global.u64 	%rd43, %rd4;
	mul.wide.s32 	%rd44, %r1, 4;
	add.s64 	%rd45, %rd43, %rd44;
	ld.global.nc.f32 	%f42, [%rd45];
	mul.f32 	%f43, %f41, %f42;
	cvta.to.global.u64 	%rd46, %rd5;
	add.s64 	%rd47, %rd46, %rd44;
	ld.global.nc.f32 	%f44, [%rd47];
	cvta.to.global.u64 	%rd48, %rd6;
	add.s64 	%rd49, %rd48, %rd44;
	ld.global.nc.f32 	%f45, [%rd49];
	mul.f32 	%f46, %f71, %f45;
	fma.rn.f32 	%f47, %f70, %f44, %f46;
	mul.f32 	%f48, %f74, %f76;
	mul.f32 	%f49, %f48, %f47;
	fma.rn.f32 	%f50, %f67, %f43, %f49;
	cvta.to.global.u64 	%rd50, %rd1;
	add.s64 	%rd51, %rd50, %rd44;
	ld.global.f32 	%f51, [%rd51];
	sub.f32 	%f52, %f51, %f50;
	st.global.f32 	[%rd51], %f52;
	mul.f32 	%f53, %f41, %f44;
	mul.f32 	%f54, %f72, %f45;
	fma.rn.f32 	%f55, %f70, %f42, %f54;
	mul.f32 	%f56, %f48, %f55;
	fma.rn.f32 	%f57, %f68, %f53, %f56;
	cvta.to.global.u64 	%rd52, %rd2;
	add.s64 	%rd53, %rd52, %rd44;
	ld.global.f32 	%f58, [%rd53];
	sub.f32 	%f59, %f58, %f57;
	st.global.f32 	[%rd53], %f59;
	mul.f32 	%f60, %f41, %f45;
	mul.f32 	%f61, %f72, %f44;
	fma.rn.f32 	%f62, %f71, %f42, %f61;
	mul.f32 	%f63, %f48, %f62;
	fma.rn.f32 	%f64, %f69, %f60, %f63;
	cvta.to.global.u64 	%rd54, %rd3;
	add.s64 	%rd55, %rd54, %rd44;
	ld.global.f32 	%f65, [%rd55];
	sub.f32 	%f66, %f65, %f64;
	st.global.f32 	[%rd55], %f66;

$L__BB0_22:
	ret;

}

`
)
