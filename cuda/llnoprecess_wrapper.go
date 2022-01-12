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

// CUDA handle for llnoprecess kernel
var llnoprecess_code cu.Function

// Stores the arguments for llnoprecess kernel invocation
type llnoprecess_args_t struct {
	arg_tx unsafe.Pointer
	arg_ty unsafe.Pointer
	arg_tz unsafe.Pointer
	arg_mx unsafe.Pointer
	arg_my unsafe.Pointer
	arg_mz unsafe.Pointer
	arg_hx unsafe.Pointer
	arg_hy unsafe.Pointer
	arg_hz unsafe.Pointer
	arg_N  int
	argptr [10]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for llnoprecess kernel invocation
var llnoprecess_args llnoprecess_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	llnoprecess_args.argptr[0] = unsafe.Pointer(&llnoprecess_args.arg_tx)
	llnoprecess_args.argptr[1] = unsafe.Pointer(&llnoprecess_args.arg_ty)
	llnoprecess_args.argptr[2] = unsafe.Pointer(&llnoprecess_args.arg_tz)
	llnoprecess_args.argptr[3] = unsafe.Pointer(&llnoprecess_args.arg_mx)
	llnoprecess_args.argptr[4] = unsafe.Pointer(&llnoprecess_args.arg_my)
	llnoprecess_args.argptr[5] = unsafe.Pointer(&llnoprecess_args.arg_mz)
	llnoprecess_args.argptr[6] = unsafe.Pointer(&llnoprecess_args.arg_hx)
	llnoprecess_args.argptr[7] = unsafe.Pointer(&llnoprecess_args.arg_hy)
	llnoprecess_args.argptr[8] = unsafe.Pointer(&llnoprecess_args.arg_hz)
	llnoprecess_args.argptr[9] = unsafe.Pointer(&llnoprecess_args.arg_N)
}

// Wrapper for llnoprecess CUDA kernel, asynchronous.
func k_llnoprecess_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, hx unsafe.Pointer, hy unsafe.Pointer, hz unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("llnoprecess")
	}

	llnoprecess_args.Lock()
	defer llnoprecess_args.Unlock()

	if llnoprecess_code == 0 {
		llnoprecess_code = fatbinLoad(llnoprecess_map, "llnoprecess")
	}

	llnoprecess_args.arg_tx = tx
	llnoprecess_args.arg_ty = ty
	llnoprecess_args.arg_tz = tz
	llnoprecess_args.arg_mx = mx
	llnoprecess_args.arg_my = my
	llnoprecess_args.arg_mz = mz
	llnoprecess_args.arg_hx = hx
	llnoprecess_args.arg_hy = hy
	llnoprecess_args.arg_hz = hz
	llnoprecess_args.arg_N = N

	args := llnoprecess_args.argptr[:]
	cu.LaunchKernel(llnoprecess_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("llnoprecess")
	}
}

// maps compute capability on PTX code for llnoprecess kernel.
var llnoprecess_map = map[int]string{0: "",
	35: llnoprecess_ptx_35}

// llnoprecess PTX code for various compute capabilities.
const (
	llnoprecess_ptx_35 = `
.version 7.5
.target sm_35
.address_size 64

	// .globl	llnoprecess

.visible .entry llnoprecess(
	.param .u64 llnoprecess_param_0,
	.param .u64 llnoprecess_param_1,
	.param .u64 llnoprecess_param_2,
	.param .u64 llnoprecess_param_3,
	.param .u64 llnoprecess_param_4,
	.param .u64 llnoprecess_param_5,
	.param .u64 llnoprecess_param_6,
	.param .u64 llnoprecess_param_7,
	.param .u64 llnoprecess_param_8,
	.param .u32 llnoprecess_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<28>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<29>;


	ld.param.u64 	%rd1, [llnoprecess_param_0];
	ld.param.u64 	%rd2, [llnoprecess_param_1];
	ld.param.u64 	%rd3, [llnoprecess_param_2];
	ld.param.u64 	%rd4, [llnoprecess_param_3];
	ld.param.u64 	%rd5, [llnoprecess_param_4];
	ld.param.u64 	%rd6, [llnoprecess_param_5];
	ld.param.u64 	%rd7, [llnoprecess_param_6];
	ld.param.u64 	%rd8, [llnoprecess_param_7];
	ld.param.u64 	%rd9, [llnoprecess_param_8];
	ld.param.u32 	%r2, [llnoprecess_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd10, %rd4;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd11;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd11;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd11;
	cvta.to.global.u64 	%rd19, %rd8;
	add.s64 	%rd20, %rd19, %rd11;
	cvta.to.global.u64 	%rd21, %rd9;
	add.s64 	%rd22, %rd21, %rd11;
	ld.global.nc.f32 	%f1, [%rd22];
	ld.global.nc.f32 	%f2, [%rd14];
	mul.f32 	%f3, %f2, %f1;
	ld.global.nc.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd16];
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	ld.global.nc.f32 	%f8, [%rd18];
	mul.f32 	%f9, %f5, %f8;
	ld.global.nc.f32 	%f10, [%rd12];
	mul.f32 	%f11, %f10, %f1;
	sub.f32 	%f12, %f9, %f11;
	mul.f32 	%f13, %f10, %f4;
	mul.f32 	%f14, %f2, %f8;
	sub.f32 	%f15, %f13, %f14;
	mul.f32 	%f16, %f2, %f15;
	mul.f32 	%f17, %f5, %f12;
	sub.f32 	%f18, %f16, %f17;
	mul.f32 	%f19, %f5, %f7;
	mul.f32 	%f20, %f10, %f15;
	sub.f32 	%f21, %f19, %f20;
	mul.f32 	%f22, %f10, %f12;
	mul.f32 	%f23, %f2, %f7;
	sub.f32 	%f24, %f22, %f23;
	neg.f32 	%f25, %f18;
	neg.f32 	%f26, %f21;
	neg.f32 	%f27, %f24;
	cvta.to.global.u64 	%rd23, %rd1;
	add.s64 	%rd24, %rd23, %rd11;
	st.global.f32 	[%rd24], %f25;
	cvta.to.global.u64 	%rd25, %rd2;
	add.s64 	%rd26, %rd25, %rd11;
	st.global.f32 	[%rd26], %f26;
	cvta.to.global.u64 	%rd27, %rd3;
	add.s64 	%rd28, %rd27, %rd11;
	st.global.f32 	[%rd28], %f27;

$L__BB0_2:
	ret;

}

`
)
