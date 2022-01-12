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

// CUDA handle for setPhi kernel
var setPhi_code cu.Function

// Stores the arguments for setPhi kernel invocation
type setPhi_args_t struct {
	arg_phi unsafe.Pointer
	arg_mx  unsafe.Pointer
	arg_my  unsafe.Pointer
	arg_Nx  int
	arg_Ny  int
	arg_Nz  int
	argptr  [6]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for setPhi kernel invocation
var setPhi_args setPhi_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	setPhi_args.argptr[0] = unsafe.Pointer(&setPhi_args.arg_phi)
	setPhi_args.argptr[1] = unsafe.Pointer(&setPhi_args.arg_mx)
	setPhi_args.argptr[2] = unsafe.Pointer(&setPhi_args.arg_my)
	setPhi_args.argptr[3] = unsafe.Pointer(&setPhi_args.arg_Nx)
	setPhi_args.argptr[4] = unsafe.Pointer(&setPhi_args.arg_Ny)
	setPhi_args.argptr[5] = unsafe.Pointer(&setPhi_args.arg_Nz)
}

// Wrapper for setPhi CUDA kernel, asynchronous.
func k_setPhi_async(phi unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, Nx int, Ny int, Nz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("setPhi")
	}

	setPhi_args.Lock()
	defer setPhi_args.Unlock()

	if setPhi_code == 0 {
		setPhi_code = fatbinLoad(setPhi_map, "setPhi")
	}

	setPhi_args.arg_phi = phi
	setPhi_args.arg_mx = mx
	setPhi_args.arg_my = my
	setPhi_args.arg_Nx = Nx
	setPhi_args.arg_Ny = Ny
	setPhi_args.arg_Nz = Nz

	args := setPhi_args.argptr[:]
	cu.LaunchKernel(setPhi_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("setPhi")
	}
}

// maps compute capability on PTX code for setPhi kernel.
var setPhi_map = map[int]string{0: "",
	35: setPhi_ptx_35}

// setPhi PTX code for various compute capabilities.
const (
	setPhi_ptx_35 = `
.version 7.5
.target sm_35
.address_size 64

	// .globl	setPhi

.visible .entry setPhi(
	.param .u64 setPhi_param_0,
	.param .u64 setPhi_param_1,
	.param .u64 setPhi_param_2,
	.param .u32 setPhi_param_3,
	.param .u32 setPhi_param_4,
	.param .u32 setPhi_param_5
)
{
	.reg .pred 	%p<16>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<12>;


	ld.param.u64 	%rd1, [setPhi_param_0];
	ld.param.u64 	%rd2, [setPhi_param_1];
	ld.param.u64 	%rd3, [setPhi_param_2];
	ld.param.u32 	%r7, [setPhi_param_3];
	ld.param.u32 	%r8, [setPhi_param_4];
	ld.param.u32 	%r9, [setPhi_param_5];
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %tid.x;
	mad.lo.s32 	%r1, %r10, %r11, %r12;
	mov.u32 	%r13, %ntid.y;
	mov.u32 	%r14, %ctaid.y;
	mov.u32 	%r15, %tid.y;
	mad.lo.s32 	%r2, %r14, %r13, %r15;
	mov.u32 	%r16, %ntid.z;
	mov.u32 	%r17, %ctaid.z;
	mov.u32 	%r18, %tid.z;
	mad.lo.s32 	%r3, %r17, %r16, %r18;
	setp.ge.s32 	%p1, %r1, %r7;
	setp.ge.s32 	%p2, %r2, %r8;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r9;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	mad.lo.s32 	%r19, %r3, %r8, %r2;
	mad.lo.s32 	%r4, %r19, %r7, %r1;
	mul.wide.s32 	%rd5, %r4, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd2;
	add.s64 	%rd8, %rd7, %rd5;
	ld.global.nc.f32 	%f7, [%rd8];
	abs.f32 	%f1, %f7;
	ld.global.nc.f32 	%f8, [%rd6];
	abs.f32 	%f2, %f8;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	setp.eq.f32 	%p7, %f2, 0f00000000;
	and.pred  	%p8, %p6, %p7;
	mov.b32 	%r5, %f7;
	mov.b32 	%r20, %f8;
	and.b32  	%r6, %r20, -2147483648;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	shr.s32 	%r25, %r5, 31;
	and.b32  	%r26, %r25, 1078530011;
	or.b32  	%r27, %r26, %r6;
	mov.b32 	%f35, %r27;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p9, %f1, 0f7F800000;
	setp.eq.f32 	%p10, %f2, 0f7F800000;
	and.pred  	%p11, %p9, %p10;
	@%p11 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	setp.lt.s32 	%p15, %r5, 0;
	selp.b32 	%r23, 1075235812, 1061752795, %p15;
	or.b32  	%r24, %r23, %r6;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	setp.lt.s32 	%p12, %r5, 0;
	min.f32 	%f9, %f2, %f1;
	max.f32 	%f10, %f2, %f1;
	div.rn.f32 	%f11, %f9, %f10;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p13, %f2, %f1;
	selp.f32 	%f29, %f28, %f26, %p13;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p12;
	mov.b32 	%r21, %f32;
	or.b32  	%r22, %r6, %r21;
	mov.b32 	%f33, %r22;
	add.f32 	%f34, %f1, %f2;
	setp.le.f32 	%p14, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p14;

$L__BB0_6:
	cvta.to.global.u64 	%rd9, %rd1;
	add.s64 	%rd11, %rd9, %rd5;
	st.global.f32 	[%rd11], %f35;

$L__BB0_7:
	ret;

}

`
)
