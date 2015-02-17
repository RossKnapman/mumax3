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

// CUDA handle for exchangedecode kernel
var exchangedecode_code cu.Function

// Stores the arguments for exchangedecode kernel invocation
type exchangedecode_args_t struct {
	arg_dst     unsafe.Pointer
	arg_aLUT2d  unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_wx      float32
	arg_wy      float32
	arg_wz      float32
	arg_Nx      int
	arg_Ny      int
	arg_Nz      int
	arg_PBC     byte
	argptr      [10]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for exchangedecode kernel invocation
var exchangedecode_args exchangedecode_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	exchangedecode_args.argptr[0] = unsafe.Pointer(&exchangedecode_args.arg_dst)
	exchangedecode_args.argptr[1] = unsafe.Pointer(&exchangedecode_args.arg_aLUT2d)
	exchangedecode_args.argptr[2] = unsafe.Pointer(&exchangedecode_args.arg_regions)
	exchangedecode_args.argptr[3] = unsafe.Pointer(&exchangedecode_args.arg_wx)
	exchangedecode_args.argptr[4] = unsafe.Pointer(&exchangedecode_args.arg_wy)
	exchangedecode_args.argptr[5] = unsafe.Pointer(&exchangedecode_args.arg_wz)
	exchangedecode_args.argptr[6] = unsafe.Pointer(&exchangedecode_args.arg_Nx)
	exchangedecode_args.argptr[7] = unsafe.Pointer(&exchangedecode_args.arg_Ny)
	exchangedecode_args.argptr[8] = unsafe.Pointer(&exchangedecode_args.arg_Nz)
	exchangedecode_args.argptr[9] = unsafe.Pointer(&exchangedecode_args.arg_PBC)
}

// Wrapper for exchangedecode CUDA kernel, asynchronous.
func k_exchangedecode_async(dst unsafe.Pointer, aLUT2d unsafe.Pointer, regions unsafe.Pointer, wx float32, wy float32, wz float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("exchangedecode")
	}

	exchangedecode_args.Lock()
	defer exchangedecode_args.Unlock()

	if exchangedecode_code == 0 {
		exchangedecode_code = fatbinLoad(exchangedecode_map, "exchangedecode")
	}

	exchangedecode_args.arg_dst = dst
	exchangedecode_args.arg_aLUT2d = aLUT2d
	exchangedecode_args.arg_regions = regions
	exchangedecode_args.arg_wx = wx
	exchangedecode_args.arg_wy = wy
	exchangedecode_args.arg_wz = wz
	exchangedecode_args.arg_Nx = Nx
	exchangedecode_args.arg_Ny = Ny
	exchangedecode_args.arg_Nz = Nz
	exchangedecode_args.arg_PBC = PBC

	args := exchangedecode_args.argptr[:]
	cu.LaunchKernel(exchangedecode_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("exchangedecode")
	}
}

// maps compute capability on PTX code for exchangedecode kernel.
var exchangedecode_map = map[int]string{0: "",
	20: exchangedecode_ptx_20,
	30: exchangedecode_ptx_30,
	35: exchangedecode_ptx_35,
	50: exchangedecode_ptx_50}

// exchangedecode PTX code for various compute capabilities.
const (
	exchangedecode_ptx_20 = `
.version 4.1
.target sm_20
.address_size 64


.visible .entry exchangedecode(
	.param .u64 exchangedecode_param_0,
	.param .u64 exchangedecode_param_1,
	.param .u64 exchangedecode_param_2,
	.param .f32 exchangedecode_param_3,
	.param .f32 exchangedecode_param_4,
	.param .f32 exchangedecode_param_5,
	.param .u32 exchangedecode_param_6,
	.param .u32 exchangedecode_param_7,
	.param .u32 exchangedecode_param_8,
	.param .u8 exchangedecode_param_9
)
{
	.reg .pred 	%p<19>;
	.reg .s16 	%rs<20>;
	.reg .s32 	%r<178>;
	.reg .f32 	%f<15>;
	.reg .s64 	%rd<40>;


	ld.param.u64 	%rd3, [exchangedecode_param_0];
	ld.param.u64 	%rd4, [exchangedecode_param_1];
	ld.param.u64 	%rd5, [exchangedecode_param_2];
	ld.param.u32 	%r55, [exchangedecode_param_6];
	ld.param.u32 	%r56, [exchangedecode_param_7];
	ld.param.u32 	%r57, [exchangedecode_param_8];
	ld.param.u8 	%rs5, [exchangedecode_param_9];
	cvta.to.global.u64 	%rd1, %rd5;
	mov.u32 	%r58, %ntid.x;
	mov.u32 	%r59, %ctaid.x;
	mov.u32 	%r60, %tid.x;
	mad.lo.s32 	%r1, %r58, %r59, %r60;
	mov.u32 	%r61, %ntid.y;
	mov.u32 	%r62, %ctaid.y;
	mov.u32 	%r63, %tid.y;
	mad.lo.s32 	%r2, %r61, %r62, %r63;
	mov.u32 	%r64, %ntid.z;
	mov.u32 	%r65, %ctaid.z;
	mov.u32 	%r66, %tid.z;
	mad.lo.s32 	%r3, %r64, %r65, %r66;
	setp.ge.s32	%p1, %r2, %r56;
	setp.ge.s32	%p2, %r1, %r55;
	or.pred  	%p3, %p2, %p1;
	setp.ge.s32	%p4, %r3, %r57;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_40;

	mul.lo.s32 	%r4, %r3, %r56;
	add.s32 	%r67, %r4, %r2;
	mul.lo.s32 	%r5, %r67, %r55;
	add.s32 	%r68, %r5, %r1;
	cvt.s64.s32	%rd2, %r68;
	add.s64 	%rd6, %rd1, %rd2;
	ld.global.u8 	%rs1, [%rd6];
	cvt.u32.u16	%r69, %rs1;
	and.b32  	%r6, %r69, 255;
	add.s32 	%r7, %r1, -1;
	and.b16  	%rs2, %rs5, 1;
	setp.eq.b16	%p6, %rs2, 1;
	@!%p6 bra 	BB0_3;
	bra.uni 	BB0_2;

BB0_2:
	rem.s32 	%r70, %r7, %r55;
	add.s32 	%r71, %r70, %r55;
	rem.s32 	%r166, %r71, %r55;
	bra.uni 	BB0_4;

BB0_3:
	mov.u32 	%r72, 0;
	max.s32 	%r166, %r7, %r72;

BB0_4:
	add.s32 	%r73, %r166, %r5;
	cvt.s64.s32	%rd7, %r73;
	add.s64 	%rd8, %rd1, %rd7;
	ld.global.u8 	%rs6, [%rd8];
	and.b16  	%rs7, %rs1, 255;
	setp.gt.u16	%p7, %rs6, %rs7;
	cvt.u32.u16	%r11, %rs6;
	@%p7 bra 	BB0_6;

	add.s32 	%r74, %r6, 1;
	mul.lo.s32 	%r75, %r74, %r6;
	shr.u32 	%r76, %r75, 31;
	add.s32 	%r77, %r75, %r76;
	shr.s32 	%r78, %r77, 1;
	add.s32 	%r167, %r78, %r11;
	bra.uni 	BB0_7;

BB0_6:
	add.s32 	%r79, %r11, 1;
	mul.lo.s32 	%r80, %r79, %r11;
	shr.u32 	%r81, %r80, 31;
	add.s32 	%r82, %r80, %r81;
	shr.s32 	%r83, %r82, 1;
	add.s32 	%r167, %r83, %r6;

BB0_7:
	cvta.to.global.u64 	%rd9, %rd4;
	mul.wide.s32 	%rd10, %r167, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.f32 	%f8, [%rd11];
	add.f32 	%f1, %f8, 0f00000000;
	add.s32 	%r15, %r1, 1;
	setp.eq.s16	%p8, %rs2, 0;
	@%p8 bra 	BB0_9;

	rem.s32 	%r84, %r15, %r55;
	add.s32 	%r85, %r84, %r55;
	rem.s32 	%r168, %r85, %r55;
	bra.uni 	BB0_10;

BB0_9:
	add.s32 	%r86, %r55, -1;
	min.s32 	%r168, %r15, %r86;

BB0_10:
	add.s32 	%r87, %r168, %r5;
	cvt.s64.s32	%rd12, %r87;
	add.s64 	%rd13, %rd1, %rd12;
	ld.global.u8 	%rs8, [%rd13];
	setp.gt.u16	%p9, %rs8, %rs7;
	cvt.u32.u16	%r19, %rs8;
	@%p9 bra 	BB0_12;

	add.s32 	%r88, %r6, 1;
	mul.lo.s32 	%r89, %r88, %r6;
	shr.u32 	%r90, %r89, 31;
	add.s32 	%r91, %r89, %r90;
	shr.s32 	%r92, %r91, 1;
	add.s32 	%r169, %r92, %r19;
	bra.uni 	BB0_13;

BB0_12:
	add.s32 	%r93, %r19, 1;
	mul.lo.s32 	%r94, %r93, %r19;
	shr.u32 	%r95, %r94, 31;
	add.s32 	%r96, %r94, %r95;
	shr.s32 	%r97, %r96, 1;
	add.s32 	%r169, %r97, %r6;

BB0_13:
	mul.wide.s32 	%rd15, %r169, 4;
	add.s64 	%rd16, %rd9, %rd15;
	ld.global.f32 	%f9, [%rd16];
	add.f32 	%f2, %f1, %f9;
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16	%p10, %rs3, 0;
	add.s32 	%r23, %r2, -1;
	@%p10 bra 	BB0_15;

	rem.s32 	%r98, %r23, %r56;
	add.s32 	%r99, %r98, %r56;
	rem.s32 	%r170, %r99, %r56;
	bra.uni 	BB0_16;

BB0_15:
	mov.u32 	%r100, 0;
	max.s32 	%r170, %r23, %r100;

BB0_16:
	add.s32 	%r101, %r170, %r4;
	mad.lo.s32 	%r102, %r101, %r55, %r1;
	cvt.s64.s32	%rd17, %r102;
	add.s64 	%rd18, %rd1, %rd17;
	ld.global.u8 	%rs10, [%rd18];
	setp.gt.u16	%p11, %rs10, %rs7;
	cvt.u32.u16	%r27, %rs10;
	@%p11 bra 	BB0_18;

	add.s32 	%r103, %r6, 1;
	mul.lo.s32 	%r104, %r103, %r6;
	shr.u32 	%r105, %r104, 31;
	add.s32 	%r106, %r104, %r105;
	shr.s32 	%r107, %r106, 1;
	add.s32 	%r171, %r107, %r27;
	bra.uni 	BB0_19;

BB0_18:
	add.s32 	%r108, %r27, 1;
	mul.lo.s32 	%r109, %r108, %r27;
	shr.u32 	%r110, %r109, 31;
	add.s32 	%r111, %r109, %r110;
	shr.s32 	%r112, %r111, 1;
	add.s32 	%r171, %r112, %r6;

BB0_19:
	mul.wide.s32 	%rd20, %r171, 4;
	add.s64 	%rd21, %rd9, %rd20;
	ld.global.f32 	%f10, [%rd21];
	add.f32 	%f3, %f2, %f10;
	add.s32 	%r31, %r2, 1;
	and.b16  	%rs12, %rs3, 255;
	setp.eq.s16	%p12, %rs12, 0;
	@%p12 bra 	BB0_21;

	rem.s32 	%r113, %r31, %r56;
	add.s32 	%r114, %r113, %r56;
	rem.s32 	%r172, %r114, %r56;
	bra.uni 	BB0_22;

BB0_21:
	add.s32 	%r115, %r56, -1;
	min.s32 	%r172, %r31, %r115;

BB0_22:
	add.s32 	%r116, %r172, %r4;
	mad.lo.s32 	%r117, %r116, %r55, %r1;
	cvt.s64.s32	%rd22, %r117;
	add.s64 	%rd23, %rd1, %rd22;
	ld.global.u8 	%rs13, [%rd23];
	setp.gt.u16	%p13, %rs13, %rs7;
	cvt.u32.u16	%r35, %rs13;
	@%p13 bra 	BB0_24;

	add.s32 	%r118, %r6, 1;
	mul.lo.s32 	%r119, %r118, %r6;
	shr.u32 	%r120, %r119, 31;
	add.s32 	%r121, %r119, %r120;
	shr.s32 	%r122, %r121, 1;
	add.s32 	%r173, %r122, %r35;
	bra.uni 	BB0_25;

BB0_24:
	add.s32 	%r123, %r35, 1;
	mul.lo.s32 	%r124, %r123, %r35;
	shr.u32 	%r125, %r124, 31;
	add.s32 	%r126, %r124, %r125;
	shr.s32 	%r127, %r126, 1;
	add.s32 	%r173, %r127, %r6;

BB0_25:
	mul.wide.s32 	%rd25, %r173, 4;
	add.s64 	%rd26, %rd9, %rd25;
	ld.global.f32 	%f11, [%rd26];
	add.f32 	%f14, %f3, %f11;
	setp.eq.s32	%p14, %r57, 1;
	@%p14 bra 	BB0_39;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16	%p15, %rs4, 0;
	add.s32 	%r39, %r3, -1;
	@%p15 bra 	BB0_28;

	rem.s32 	%r132, %r39, %r57;
	add.s32 	%r133, %r132, %r57;
	rem.s32 	%r174, %r133, %r57;
	bra.uni 	BB0_29;

BB0_28:
	mov.u32 	%r134, 0;
	max.s32 	%r174, %r39, %r134;

BB0_29:
	mad.lo.s32 	%r135, %r174, %r56, %r2;
	mad.lo.s32 	%r136, %r135, %r55, %r1;
	cvt.s64.s32	%rd27, %r136;
	add.s64 	%rd28, %rd1, %rd27;
	ld.global.u8 	%rs15, [%rd28];
	setp.gt.u16	%p16, %rs15, %rs7;
	cvt.u32.u16	%r43, %rs15;
	@%p16 bra 	BB0_31;

	add.s32 	%r137, %r6, 1;
	mul.lo.s32 	%r138, %r137, %r6;
	shr.u32 	%r139, %r138, 31;
	add.s32 	%r140, %r138, %r139;
	shr.s32 	%r141, %r140, 1;
	add.s32 	%r175, %r141, %r43;
	bra.uni 	BB0_32;

BB0_31:
	add.s32 	%r142, %r43, 1;
	mul.lo.s32 	%r143, %r142, %r43;
	shr.u32 	%r144, %r143, 31;
	add.s32 	%r145, %r143, %r144;
	shr.s32 	%r146, %r145, 1;
	add.s32 	%r175, %r146, %r6;

BB0_32:
	mul.wide.s32 	%rd30, %r175, 4;
	add.s64 	%rd31, %rd9, %rd30;
	ld.global.f32 	%f12, [%rd31];
	add.f32 	%f5, %f14, %f12;
	add.s32 	%r47, %r3, 1;
	and.b16  	%rs17, %rs4, 255;
	setp.eq.s16	%p17, %rs17, 0;
	@%p17 bra 	BB0_34;

	rem.s32 	%r151, %r47, %r57;
	add.s32 	%r152, %r151, %r57;
	rem.s32 	%r176, %r152, %r57;
	bra.uni 	BB0_35;

BB0_34:
	add.s32 	%r153, %r57, -1;
	min.s32 	%r176, %r47, %r153;

BB0_35:
	mad.lo.s32 	%r154, %r176, %r56, %r2;
	mad.lo.s32 	%r155, %r154, %r55, %r1;
	cvt.s64.s32	%rd32, %r155;
	add.s64 	%rd33, %rd1, %rd32;
	ld.global.u8 	%rs18, [%rd33];
	setp.gt.u16	%p18, %rs18, %rs7;
	cvt.u32.u16	%r51, %rs18;
	@%p18 bra 	BB0_37;

	add.s32 	%r156, %r6, 1;
	mul.lo.s32 	%r157, %r156, %r6;
	shr.u32 	%r158, %r157, 31;
	add.s32 	%r159, %r157, %r158;
	shr.s32 	%r160, %r159, 1;
	add.s32 	%r177, %r160, %r51;
	bra.uni 	BB0_38;

BB0_37:
	add.s32 	%r161, %r51, 1;
	mul.lo.s32 	%r162, %r161, %r51;
	shr.u32 	%r163, %r162, 31;
	add.s32 	%r164, %r162, %r163;
	shr.s32 	%r165, %r164, 1;
	add.s32 	%r177, %r165, %r6;

BB0_38:
	mul.wide.s32 	%rd35, %r177, 4;
	add.s64 	%rd36, %rd9, %rd35;
	ld.global.f32 	%f13, [%rd36];
	add.f32 	%f14, %f5, %f13;

BB0_39:
	cvta.to.global.u64 	%rd37, %rd3;
	shl.b64 	%rd38, %rd2, 2;
	add.s64 	%rd39, %rd37, %rd38;
	st.global.f32 	[%rd39], %f14;

BB0_40:
	ret;
}


`
	exchangedecode_ptx_30 = `
.version 4.1
.target sm_30
.address_size 64


.visible .entry exchangedecode(
	.param .u64 exchangedecode_param_0,
	.param .u64 exchangedecode_param_1,
	.param .u64 exchangedecode_param_2,
	.param .f32 exchangedecode_param_3,
	.param .f32 exchangedecode_param_4,
	.param .f32 exchangedecode_param_5,
	.param .u32 exchangedecode_param_6,
	.param .u32 exchangedecode_param_7,
	.param .u32 exchangedecode_param_8,
	.param .u8 exchangedecode_param_9
)
{
	.reg .pred 	%p<19>;
	.reg .s16 	%rs<20>;
	.reg .s32 	%r<170>;
	.reg .f32 	%f<15>;
	.reg .s64 	%rd<46>;


	ld.param.u64 	%rd2, [exchangedecode_param_0];
	ld.param.u64 	%rd3, [exchangedecode_param_1];
	ld.param.u64 	%rd4, [exchangedecode_param_2];
	ld.param.u32 	%r55, [exchangedecode_param_6];
	ld.param.u32 	%r56, [exchangedecode_param_7];
	ld.param.u32 	%r57, [exchangedecode_param_8];
	ld.param.u8 	%rs5, [exchangedecode_param_9];
	mov.u32 	%r58, %ntid.x;
	mov.u32 	%r59, %ctaid.x;
	mov.u32 	%r60, %tid.x;
	mad.lo.s32 	%r1, %r58, %r59, %r60;
	mov.u32 	%r61, %ntid.y;
	mov.u32 	%r62, %ctaid.y;
	mov.u32 	%r63, %tid.y;
	mad.lo.s32 	%r2, %r61, %r62, %r63;
	mov.u32 	%r64, %ntid.z;
	mov.u32 	%r65, %ctaid.z;
	mov.u32 	%r66, %tid.z;
	mad.lo.s32 	%r3, %r64, %r65, %r66;
	setp.ge.s32	%p1, %r2, %r56;
	setp.ge.s32	%p2, %r1, %r55;
	or.pred  	%p3, %p2, %p1;
	setp.ge.s32	%p4, %r3, %r57;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_40;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.lo.s32 	%r4, %r3, %r56;
	add.s32 	%r67, %r4, %r2;
	mul.lo.s32 	%r5, %r67, %r55;
	add.s32 	%r68, %r5, %r1;
	cvt.s64.s32	%rd1, %r68;
	add.s64 	%rd6, %rd5, %rd1;
	ld.global.u8 	%rs1, [%rd6];
	cvt.u32.u16	%r69, %rs1;
	and.b32  	%r6, %r69, 255;
	add.s32 	%r7, %r1, -1;
	and.b16  	%rs2, %rs5, 1;
	setp.eq.b16	%p6, %rs2, 1;
	@!%p6 bra 	BB0_3;
	bra.uni 	BB0_2;

BB0_2:
	rem.s32 	%r70, %r7, %r55;
	add.s32 	%r71, %r70, %r55;
	rem.s32 	%r158, %r71, %r55;
	bra.uni 	BB0_4;

BB0_3:
	mov.u32 	%r72, 0;
	max.s32 	%r158, %r7, %r72;

BB0_4:
	add.s32 	%r73, %r158, %r5;
	cvt.s64.s32	%rd8, %r73;
	add.s64 	%rd9, %rd5, %rd8;
	ld.global.u8 	%rs6, [%rd9];
	and.b16  	%rs7, %rs1, 255;
	setp.gt.u16	%p7, %rs6, %rs7;
	cvt.u32.u16	%r11, %rs6;
	@%p7 bra 	BB0_6;

	add.s32 	%r74, %r6, 1;
	mul.lo.s32 	%r75, %r74, %r6;
	shr.u32 	%r76, %r75, 31;
	add.s32 	%r77, %r75, %r76;
	shr.s32 	%r78, %r77, 1;
	add.s32 	%r159, %r78, %r11;
	bra.uni 	BB0_7;

BB0_6:
	add.s32 	%r79, %r11, 1;
	mul.lo.s32 	%r80, %r79, %r11;
	shr.u32 	%r81, %r80, 31;
	add.s32 	%r82, %r80, %r81;
	shr.s32 	%r83, %r82, 1;
	add.s32 	%r159, %r83, %r6;

BB0_7:
	cvta.to.global.u64 	%rd10, %rd3;
	mul.wide.s32 	%rd11, %r159, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f8, [%rd12];
	add.f32 	%f1, %f8, 0f00000000;
	add.s32 	%r15, %r1, 1;
	setp.eq.s16	%p8, %rs2, 0;
	@%p8 bra 	BB0_9;

	rem.s32 	%r84, %r15, %r55;
	add.s32 	%r85, %r84, %r55;
	rem.s32 	%r160, %r85, %r55;
	bra.uni 	BB0_10;

BB0_9:
	add.s32 	%r86, %r55, -1;
	min.s32 	%r160, %r15, %r86;

BB0_10:
	add.s32 	%r87, %r160, %r5;
	cvt.s64.s32	%rd14, %r87;
	add.s64 	%rd15, %rd5, %rd14;
	ld.global.u8 	%rs8, [%rd15];
	setp.gt.u16	%p9, %rs8, %rs7;
	cvt.u32.u16	%r19, %rs8;
	@%p9 bra 	BB0_12;

	add.s32 	%r88, %r6, 1;
	mul.lo.s32 	%r89, %r88, %r6;
	shr.u32 	%r90, %r89, 31;
	add.s32 	%r91, %r89, %r90;
	shr.s32 	%r92, %r91, 1;
	add.s32 	%r161, %r92, %r19;
	bra.uni 	BB0_13;

BB0_12:
	add.s32 	%r93, %r19, 1;
	mul.lo.s32 	%r94, %r93, %r19;
	shr.u32 	%r95, %r94, 31;
	add.s32 	%r96, %r94, %r95;
	shr.s32 	%r97, %r96, 1;
	add.s32 	%r161, %r97, %r6;

BB0_13:
	mul.wide.s32 	%rd17, %r161, 4;
	add.s64 	%rd18, %rd10, %rd17;
	ld.global.f32 	%f9, [%rd18];
	add.f32 	%f2, %f1, %f9;
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16	%p10, %rs3, 0;
	add.s32 	%r23, %r2, -1;
	@%p10 bra 	BB0_15;

	rem.s32 	%r98, %r23, %r56;
	add.s32 	%r99, %r98, %r56;
	rem.s32 	%r162, %r99, %r56;
	bra.uni 	BB0_16;

BB0_15:
	mov.u32 	%r100, 0;
	max.s32 	%r162, %r23, %r100;

BB0_16:
	add.s32 	%r101, %r162, %r4;
	mad.lo.s32 	%r102, %r101, %r55, %r1;
	cvt.s64.s32	%rd20, %r102;
	add.s64 	%rd21, %rd5, %rd20;
	ld.global.u8 	%rs10, [%rd21];
	setp.gt.u16	%p11, %rs10, %rs7;
	cvt.u32.u16	%r27, %rs10;
	@%p11 bra 	BB0_18;

	add.s32 	%r103, %r6, 1;
	mul.lo.s32 	%r104, %r103, %r6;
	shr.u32 	%r105, %r104, 31;
	add.s32 	%r106, %r104, %r105;
	shr.s32 	%r107, %r106, 1;
	add.s32 	%r163, %r107, %r27;
	bra.uni 	BB0_19;

BB0_18:
	add.s32 	%r108, %r27, 1;
	mul.lo.s32 	%r109, %r108, %r27;
	shr.u32 	%r110, %r109, 31;
	add.s32 	%r111, %r109, %r110;
	shr.s32 	%r112, %r111, 1;
	add.s32 	%r163, %r112, %r6;

BB0_19:
	mul.wide.s32 	%rd23, %r163, 4;
	add.s64 	%rd24, %rd10, %rd23;
	ld.global.f32 	%f10, [%rd24];
	add.f32 	%f3, %f2, %f10;
	add.s32 	%r31, %r2, 1;
	and.b16  	%rs12, %rs3, 255;
	setp.eq.s16	%p12, %rs12, 0;
	@%p12 bra 	BB0_21;

	rem.s32 	%r113, %r31, %r56;
	add.s32 	%r114, %r113, %r56;
	rem.s32 	%r164, %r114, %r56;
	bra.uni 	BB0_22;

BB0_21:
	add.s32 	%r115, %r56, -1;
	min.s32 	%r164, %r31, %r115;

BB0_22:
	add.s32 	%r116, %r164, %r4;
	mad.lo.s32 	%r117, %r116, %r55, %r1;
	cvt.s64.s32	%rd26, %r117;
	add.s64 	%rd27, %rd5, %rd26;
	ld.global.u8 	%rs13, [%rd27];
	setp.gt.u16	%p13, %rs13, %rs7;
	cvt.u32.u16	%r35, %rs13;
	@%p13 bra 	BB0_24;

	add.s32 	%r118, %r6, 1;
	mul.lo.s32 	%r119, %r118, %r6;
	shr.u32 	%r120, %r119, 31;
	add.s32 	%r121, %r119, %r120;
	shr.s32 	%r122, %r121, 1;
	add.s32 	%r165, %r122, %r35;
	bra.uni 	BB0_25;

BB0_24:
	add.s32 	%r123, %r35, 1;
	mul.lo.s32 	%r124, %r123, %r35;
	shr.u32 	%r125, %r124, 31;
	add.s32 	%r126, %r124, %r125;
	shr.s32 	%r127, %r126, 1;
	add.s32 	%r165, %r127, %r6;

BB0_25:
	mul.wide.s32 	%rd29, %r165, 4;
	add.s64 	%rd30, %rd10, %rd29;
	ld.global.f32 	%f11, [%rd30];
	add.f32 	%f14, %f3, %f11;
	setp.eq.s32	%p14, %r57, 1;
	@%p14 bra 	BB0_39;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16	%p15, %rs4, 0;
	add.s32 	%r39, %r3, -1;
	@%p15 bra 	BB0_28;

	rem.s32 	%r128, %r39, %r57;
	add.s32 	%r129, %r128, %r57;
	rem.s32 	%r166, %r129, %r57;
	bra.uni 	BB0_29;

BB0_28:
	mov.u32 	%r130, 0;
	max.s32 	%r166, %r39, %r130;

BB0_29:
	mad.lo.s32 	%r131, %r166, %r56, %r2;
	mad.lo.s32 	%r132, %r131, %r55, %r1;
	cvt.s64.s32	%rd32, %r132;
	add.s64 	%rd33, %rd5, %rd32;
	ld.global.u8 	%rs15, [%rd33];
	setp.gt.u16	%p16, %rs15, %rs7;
	cvt.u32.u16	%r43, %rs15;
	@%p16 bra 	BB0_31;

	add.s32 	%r133, %r6, 1;
	mul.lo.s32 	%r134, %r133, %r6;
	shr.u32 	%r135, %r134, 31;
	add.s32 	%r136, %r134, %r135;
	shr.s32 	%r137, %r136, 1;
	add.s32 	%r167, %r137, %r43;
	bra.uni 	BB0_32;

BB0_31:
	add.s32 	%r138, %r43, 1;
	mul.lo.s32 	%r139, %r138, %r43;
	shr.u32 	%r140, %r139, 31;
	add.s32 	%r141, %r139, %r140;
	shr.s32 	%r142, %r141, 1;
	add.s32 	%r167, %r142, %r6;

BB0_32:
	mul.wide.s32 	%rd35, %r167, 4;
	add.s64 	%rd36, %rd10, %rd35;
	ld.global.f32 	%f12, [%rd36];
	add.f32 	%f5, %f14, %f12;
	add.s32 	%r47, %r3, 1;
	and.b16  	%rs17, %rs4, 255;
	setp.eq.s16	%p17, %rs17, 0;
	@%p17 bra 	BB0_34;

	rem.s32 	%r143, %r47, %r57;
	add.s32 	%r144, %r143, %r57;
	rem.s32 	%r168, %r144, %r57;
	bra.uni 	BB0_35;

BB0_34:
	add.s32 	%r145, %r57, -1;
	min.s32 	%r168, %r47, %r145;

BB0_35:
	mad.lo.s32 	%r146, %r168, %r56, %r2;
	mad.lo.s32 	%r147, %r146, %r55, %r1;
	cvt.s64.s32	%rd38, %r147;
	add.s64 	%rd39, %rd5, %rd38;
	ld.global.u8 	%rs18, [%rd39];
	setp.gt.u16	%p18, %rs18, %rs7;
	cvt.u32.u16	%r51, %rs18;
	@%p18 bra 	BB0_37;

	add.s32 	%r148, %r6, 1;
	mul.lo.s32 	%r149, %r148, %r6;
	shr.u32 	%r150, %r149, 31;
	add.s32 	%r151, %r149, %r150;
	shr.s32 	%r152, %r151, 1;
	add.s32 	%r169, %r152, %r51;
	bra.uni 	BB0_38;

BB0_37:
	add.s32 	%r153, %r51, 1;
	mul.lo.s32 	%r154, %r153, %r51;
	shr.u32 	%r155, %r154, 31;
	add.s32 	%r156, %r154, %r155;
	shr.s32 	%r157, %r156, 1;
	add.s32 	%r169, %r157, %r6;

BB0_38:
	mul.wide.s32 	%rd41, %r169, 4;
	add.s64 	%rd42, %rd10, %rd41;
	ld.global.f32 	%f13, [%rd42];
	add.f32 	%f14, %f5, %f13;

BB0_39:
	cvta.to.global.u64 	%rd43, %rd2;
	shl.b64 	%rd44, %rd1, 2;
	add.s64 	%rd45, %rd43, %rd44;
	st.global.f32 	[%rd45], %f14;

BB0_40:
	ret;
}


`
	exchangedecode_ptx_35 = `
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

.visible .entry exchangedecode(
	.param .u64 exchangedecode_param_0,
	.param .u64 exchangedecode_param_1,
	.param .u64 exchangedecode_param_2,
	.param .f32 exchangedecode_param_3,
	.param .f32 exchangedecode_param_4,
	.param .f32 exchangedecode_param_5,
	.param .u32 exchangedecode_param_6,
	.param .u32 exchangedecode_param_7,
	.param .u32 exchangedecode_param_8,
	.param .u8 exchangedecode_param_9
)
{
	.reg .pred 	%p<19>;
	.reg .s16 	%rs<26>;
	.reg .s32 	%r<176>;
	.reg .f32 	%f<15>;
	.reg .s64 	%rd<35>;


	ld.param.u64 	%rd5, [exchangedecode_param_0];
	ld.param.u64 	%rd6, [exchangedecode_param_1];
	ld.param.u64 	%rd7, [exchangedecode_param_2];
	ld.param.u32 	%r55, [exchangedecode_param_6];
	ld.param.u32 	%r56, [exchangedecode_param_7];
	ld.param.u32 	%r57, [exchangedecode_param_8];
	ld.param.u8 	%rs5, [exchangedecode_param_9];
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd7;
	mov.u32 	%r58, %ntid.x;
	mov.u32 	%r59, %ctaid.x;
	mov.u32 	%r60, %tid.x;
	mad.lo.s32 	%r1, %r58, %r59, %r60;
	mov.u32 	%r61, %ntid.y;
	mov.u32 	%r62, %ctaid.y;
	mov.u32 	%r63, %tid.y;
	mad.lo.s32 	%r2, %r61, %r62, %r63;
	mov.u32 	%r64, %ntid.z;
	mov.u32 	%r65, %ctaid.z;
	mov.u32 	%r66, %tid.z;
	mad.lo.s32 	%r3, %r64, %r65, %r66;
	setp.ge.s32	%p1, %r2, %r56;
	setp.ge.s32	%p2, %r1, %r55;
	or.pred  	%p3, %p2, %p1;
	setp.ge.s32	%p4, %r3, %r57;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB5_40;

	cvta.to.global.u64 	%rd3, %rd5;
	mul.lo.s32 	%r4, %r3, %r56;
	add.s32 	%r67, %r4, %r2;
	mul.lo.s32 	%r5, %r67, %r55;
	add.s32 	%r68, %r5, %r1;
	cvt.s64.s32	%rd4, %r68;
	add.s64 	%rd8, %rd2, %rd4;
	ld.global.nc.u8 	%rs1, [%rd8];
	cvt.u32.u16	%r69, %rs1;
	and.b32  	%r6, %r69, 255;
	add.s32 	%r7, %r1, -1;
	and.b16  	%rs2, %rs5, 1;
	setp.eq.b16	%p6, %rs2, 1;
	@!%p6 bra 	BB5_3;
	bra.uni 	BB5_2;

BB5_2:
	rem.s32 	%r70, %r7, %r55;
	add.s32 	%r71, %r70, %r55;
	rem.s32 	%r164, %r71, %r55;
	bra.uni 	BB5_4;

BB5_3:
	mov.u32 	%r72, 0;
	max.s32 	%r164, %r7, %r72;

BB5_4:
	add.s32 	%r73, %r164, %r5;
	cvt.s64.s32	%rd9, %r73;
	add.s64 	%rd10, %rd2, %rd9;
	ld.global.nc.u8 	%rs6, [%rd10];
	and.b16  	%rs7, %rs6, 255;
	and.b16  	%rs8, %rs1, 255;
	setp.gt.u16	%p7, %rs7, %rs8;
	cvt.u32.u16	%r74, %rs6;
	and.b32  	%r11, %r74, 255;
	@%p7 bra 	BB5_6;

	add.s32 	%r75, %r6, 1;
	mul.lo.s32 	%r76, %r75, %r6;
	shr.u32 	%r77, %r76, 31;
	add.s32 	%r78, %r76, %r77;
	shr.s32 	%r79, %r78, 1;
	add.s32 	%r165, %r79, %r11;
	bra.uni 	BB5_7;

BB5_6:
	add.s32 	%r80, %r11, 1;
	mul.lo.s32 	%r81, %r80, %r11;
	shr.u32 	%r82, %r81, 31;
	add.s32 	%r83, %r81, %r82;
	shr.s32 	%r84, %r83, 1;
	add.s32 	%r165, %r84, %r6;

BB5_7:
	mul.wide.s32 	%rd11, %r165, 4;
	add.s64 	%rd12, %rd1, %rd11;
	ld.global.nc.f32 	%f8, [%rd12];
	add.f32 	%f1, %f8, 0f00000000;
	add.s32 	%r15, %r1, 1;
	setp.eq.s16	%p8, %rs2, 0;
	@%p8 bra 	BB5_9;

	rem.s32 	%r85, %r15, %r55;
	add.s32 	%r86, %r85, %r55;
	rem.s32 	%r166, %r86, %r55;
	bra.uni 	BB5_10;

BB5_9:
	add.s32 	%r87, %r55, -1;
	min.s32 	%r166, %r15, %r87;

BB5_10:
	add.s32 	%r88, %r166, %r5;
	cvt.s64.s32	%rd13, %r88;
	add.s64 	%rd14, %rd2, %rd13;
	ld.global.nc.u8 	%rs9, [%rd14];
	and.b16  	%rs10, %rs9, 255;
	setp.gt.u16	%p9, %rs10, %rs8;
	cvt.u32.u16	%r89, %rs9;
	and.b32  	%r19, %r89, 255;
	@%p9 bra 	BB5_12;

	add.s32 	%r90, %r6, 1;
	mul.lo.s32 	%r91, %r90, %r6;
	shr.u32 	%r92, %r91, 31;
	add.s32 	%r93, %r91, %r92;
	shr.s32 	%r94, %r93, 1;
	add.s32 	%r167, %r94, %r19;
	bra.uni 	BB5_13;

BB5_12:
	add.s32 	%r95, %r19, 1;
	mul.lo.s32 	%r96, %r95, %r19;
	shr.u32 	%r97, %r96, 31;
	add.s32 	%r98, %r96, %r97;
	shr.s32 	%r99, %r98, 1;
	add.s32 	%r167, %r99, %r6;

BB5_13:
	mul.wide.s32 	%rd15, %r167, 4;
	add.s64 	%rd16, %rd1, %rd15;
	ld.global.nc.f32 	%f9, [%rd16];
	add.f32 	%f2, %f1, %f9;
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16	%p10, %rs3, 0;
	add.s32 	%r23, %r2, -1;
	@%p10 bra 	BB5_15;

	rem.s32 	%r100, %r23, %r56;
	add.s32 	%r101, %r100, %r56;
	rem.s32 	%r168, %r101, %r56;
	bra.uni 	BB5_16;

BB5_15:
	mov.u32 	%r102, 0;
	max.s32 	%r168, %r23, %r102;

BB5_16:
	add.s32 	%r103, %r168, %r4;
	mad.lo.s32 	%r104, %r103, %r55, %r1;
	cvt.s64.s32	%rd17, %r104;
	add.s64 	%rd18, %rd2, %rd17;
	ld.global.nc.u8 	%rs12, [%rd18];
	and.b16  	%rs13, %rs12, 255;
	setp.gt.u16	%p11, %rs13, %rs8;
	cvt.u32.u16	%r105, %rs12;
	and.b32  	%r27, %r105, 255;
	@%p11 bra 	BB5_18;

	add.s32 	%r106, %r6, 1;
	mul.lo.s32 	%r107, %r106, %r6;
	shr.u32 	%r108, %r107, 31;
	add.s32 	%r109, %r107, %r108;
	shr.s32 	%r110, %r109, 1;
	add.s32 	%r169, %r110, %r27;
	bra.uni 	BB5_19;

BB5_18:
	add.s32 	%r111, %r27, 1;
	mul.lo.s32 	%r112, %r111, %r27;
	shr.u32 	%r113, %r112, 31;
	add.s32 	%r114, %r112, %r113;
	shr.s32 	%r115, %r114, 1;
	add.s32 	%r169, %r115, %r6;

BB5_19:
	mul.wide.s32 	%rd19, %r169, 4;
	add.s64 	%rd20, %rd1, %rd19;
	ld.global.nc.f32 	%f10, [%rd20];
	add.f32 	%f3, %f2, %f10;
	add.s32 	%r31, %r2, 1;
	and.b16  	%rs15, %rs3, 255;
	setp.eq.s16	%p12, %rs15, 0;
	@%p12 bra 	BB5_21;

	rem.s32 	%r116, %r31, %r56;
	add.s32 	%r117, %r116, %r56;
	rem.s32 	%r170, %r117, %r56;
	bra.uni 	BB5_22;

BB5_21:
	add.s32 	%r118, %r56, -1;
	min.s32 	%r170, %r31, %r118;

BB5_22:
	add.s32 	%r119, %r170, %r4;
	mad.lo.s32 	%r120, %r119, %r55, %r1;
	cvt.s64.s32	%rd21, %r120;
	add.s64 	%rd22, %rd2, %rd21;
	ld.global.nc.u8 	%rs16, [%rd22];
	and.b16  	%rs17, %rs16, 255;
	setp.gt.u16	%p13, %rs17, %rs8;
	cvt.u32.u16	%r121, %rs16;
	and.b32  	%r35, %r121, 255;
	@%p13 bra 	BB5_24;

	add.s32 	%r122, %r6, 1;
	mul.lo.s32 	%r123, %r122, %r6;
	shr.u32 	%r124, %r123, 31;
	add.s32 	%r125, %r123, %r124;
	shr.s32 	%r126, %r125, 1;
	add.s32 	%r171, %r126, %r35;
	bra.uni 	BB5_25;

BB5_24:
	add.s32 	%r127, %r35, 1;
	mul.lo.s32 	%r128, %r127, %r35;
	shr.u32 	%r129, %r128, 31;
	add.s32 	%r130, %r128, %r129;
	shr.s32 	%r131, %r130, 1;
	add.s32 	%r171, %r131, %r6;

BB5_25:
	mul.wide.s32 	%rd23, %r171, 4;
	add.s64 	%rd24, %rd1, %rd23;
	ld.global.nc.f32 	%f11, [%rd24];
	add.f32 	%f14, %f3, %f11;
	setp.eq.s32	%p14, %r57, 1;
	@%p14 bra 	BB5_39;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16	%p15, %rs4, 0;
	add.s32 	%r39, %r3, -1;
	@%p15 bra 	BB5_28;

	rem.s32 	%r132, %r39, %r57;
	add.s32 	%r133, %r132, %r57;
	rem.s32 	%r172, %r133, %r57;
	bra.uni 	BB5_29;

BB5_28:
	mov.u32 	%r134, 0;
	max.s32 	%r172, %r39, %r134;

BB5_29:
	mad.lo.s32 	%r135, %r172, %r56, %r2;
	mad.lo.s32 	%r136, %r135, %r55, %r1;
	cvt.s64.s32	%rd25, %r136;
	add.s64 	%rd26, %rd2, %rd25;
	ld.global.nc.u8 	%rs19, [%rd26];
	and.b16  	%rs20, %rs19, 255;
	setp.gt.u16	%p16, %rs20, %rs8;
	cvt.u32.u16	%r137, %rs19;
	and.b32  	%r43, %r137, 255;
	@%p16 bra 	BB5_31;

	add.s32 	%r138, %r6, 1;
	mul.lo.s32 	%r139, %r138, %r6;
	shr.u32 	%r140, %r139, 31;
	add.s32 	%r141, %r139, %r140;
	shr.s32 	%r142, %r141, 1;
	add.s32 	%r173, %r142, %r43;
	bra.uni 	BB5_32;

BB5_31:
	add.s32 	%r143, %r43, 1;
	mul.lo.s32 	%r144, %r143, %r43;
	shr.u32 	%r145, %r144, 31;
	add.s32 	%r146, %r144, %r145;
	shr.s32 	%r147, %r146, 1;
	add.s32 	%r173, %r147, %r6;

BB5_32:
	mul.wide.s32 	%rd27, %r173, 4;
	add.s64 	%rd28, %rd1, %rd27;
	ld.global.nc.f32 	%f12, [%rd28];
	add.f32 	%f5, %f14, %f12;
	add.s32 	%r47, %r3, 1;
	and.b16  	%rs22, %rs4, 255;
	setp.eq.s16	%p17, %rs22, 0;
	@%p17 bra 	BB5_34;

	rem.s32 	%r148, %r47, %r57;
	add.s32 	%r149, %r148, %r57;
	rem.s32 	%r174, %r149, %r57;
	bra.uni 	BB5_35;

BB5_34:
	add.s32 	%r150, %r57, -1;
	min.s32 	%r174, %r47, %r150;

BB5_35:
	mad.lo.s32 	%r151, %r174, %r56, %r2;
	mad.lo.s32 	%r152, %r151, %r55, %r1;
	cvt.s64.s32	%rd29, %r152;
	add.s64 	%rd30, %rd2, %rd29;
	ld.global.nc.u8 	%rs23, [%rd30];
	and.b16  	%rs24, %rs23, 255;
	setp.gt.u16	%p18, %rs24, %rs8;
	cvt.u32.u16	%r153, %rs23;
	and.b32  	%r51, %r153, 255;
	@%p18 bra 	BB5_37;

	add.s32 	%r154, %r6, 1;
	mul.lo.s32 	%r155, %r154, %r6;
	shr.u32 	%r156, %r155, 31;
	add.s32 	%r157, %r155, %r156;
	shr.s32 	%r158, %r157, 1;
	add.s32 	%r175, %r158, %r51;
	bra.uni 	BB5_38;

BB5_37:
	add.s32 	%r159, %r51, 1;
	mul.lo.s32 	%r160, %r159, %r51;
	shr.u32 	%r161, %r160, 31;
	add.s32 	%r162, %r160, %r161;
	shr.s32 	%r163, %r162, 1;
	add.s32 	%r175, %r163, %r6;

BB5_38:
	mul.wide.s32 	%rd31, %r175, 4;
	add.s64 	%rd32, %rd1, %rd31;
	ld.global.nc.f32 	%f13, [%rd32];
	add.f32 	%f14, %f5, %f13;

BB5_39:
	shl.b64 	%rd33, %rd4, 2;
	add.s64 	%rd34, %rd3, %rd33;
	st.global.f32 	[%rd34], %f14;

BB5_40:
	ret;
}


`
	exchangedecode_ptx_50 = `
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

.visible .entry exchangedecode(
	.param .u64 exchangedecode_param_0,
	.param .u64 exchangedecode_param_1,
	.param .u64 exchangedecode_param_2,
	.param .f32 exchangedecode_param_3,
	.param .f32 exchangedecode_param_4,
	.param .f32 exchangedecode_param_5,
	.param .u32 exchangedecode_param_6,
	.param .u32 exchangedecode_param_7,
	.param .u32 exchangedecode_param_8,
	.param .u8 exchangedecode_param_9
)
{
	.reg .pred 	%p<19>;
	.reg .s16 	%rs<26>;
	.reg .s32 	%r<176>;
	.reg .f32 	%f<15>;
	.reg .s64 	%rd<35>;


	ld.param.u64 	%rd5, [exchangedecode_param_0];
	ld.param.u64 	%rd6, [exchangedecode_param_1];
	ld.param.u64 	%rd7, [exchangedecode_param_2];
	ld.param.u32 	%r55, [exchangedecode_param_6];
	ld.param.u32 	%r56, [exchangedecode_param_7];
	ld.param.u32 	%r57, [exchangedecode_param_8];
	ld.param.u8 	%rs5, [exchangedecode_param_9];
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd7;
	mov.u32 	%r58, %ntid.x;
	mov.u32 	%r59, %ctaid.x;
	mov.u32 	%r60, %tid.x;
	mad.lo.s32 	%r1, %r58, %r59, %r60;
	mov.u32 	%r61, %ntid.y;
	mov.u32 	%r62, %ctaid.y;
	mov.u32 	%r63, %tid.y;
	mad.lo.s32 	%r2, %r61, %r62, %r63;
	mov.u32 	%r64, %ntid.z;
	mov.u32 	%r65, %ctaid.z;
	mov.u32 	%r66, %tid.z;
	mad.lo.s32 	%r3, %r64, %r65, %r66;
	setp.ge.s32	%p1, %r2, %r56;
	setp.ge.s32	%p2, %r1, %r55;
	or.pred  	%p3, %p2, %p1;
	setp.ge.s32	%p4, %r3, %r57;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB5_40;

	cvta.to.global.u64 	%rd3, %rd5;
	mul.lo.s32 	%r4, %r3, %r56;
	add.s32 	%r67, %r4, %r2;
	mul.lo.s32 	%r5, %r67, %r55;
	add.s32 	%r68, %r5, %r1;
	cvt.s64.s32	%rd4, %r68;
	add.s64 	%rd8, %rd2, %rd4;
	ld.global.nc.u8 	%rs1, [%rd8];
	cvt.u32.u16	%r69, %rs1;
	and.b32  	%r6, %r69, 255;
	add.s32 	%r7, %r1, -1;
	and.b16  	%rs2, %rs5, 1;
	setp.eq.b16	%p6, %rs2, 1;
	@!%p6 bra 	BB5_3;
	bra.uni 	BB5_2;

BB5_2:
	rem.s32 	%r70, %r7, %r55;
	add.s32 	%r71, %r70, %r55;
	rem.s32 	%r164, %r71, %r55;
	bra.uni 	BB5_4;

BB5_3:
	mov.u32 	%r72, 0;
	max.s32 	%r164, %r7, %r72;

BB5_4:
	add.s32 	%r73, %r164, %r5;
	cvt.s64.s32	%rd9, %r73;
	add.s64 	%rd10, %rd2, %rd9;
	ld.global.nc.u8 	%rs6, [%rd10];
	and.b16  	%rs7, %rs6, 255;
	and.b16  	%rs8, %rs1, 255;
	setp.gt.u16	%p7, %rs7, %rs8;
	cvt.u32.u16	%r74, %rs6;
	and.b32  	%r11, %r74, 255;
	@%p7 bra 	BB5_6;

	add.s32 	%r75, %r6, 1;
	mul.lo.s32 	%r76, %r75, %r6;
	shr.u32 	%r77, %r76, 31;
	add.s32 	%r78, %r76, %r77;
	shr.s32 	%r79, %r78, 1;
	add.s32 	%r165, %r79, %r11;
	bra.uni 	BB5_7;

BB5_6:
	add.s32 	%r80, %r11, 1;
	mul.lo.s32 	%r81, %r80, %r11;
	shr.u32 	%r82, %r81, 31;
	add.s32 	%r83, %r81, %r82;
	shr.s32 	%r84, %r83, 1;
	add.s32 	%r165, %r84, %r6;

BB5_7:
	mul.wide.s32 	%rd11, %r165, 4;
	add.s64 	%rd12, %rd1, %rd11;
	ld.global.nc.f32 	%f8, [%rd12];
	add.f32 	%f1, %f8, 0f00000000;
	add.s32 	%r15, %r1, 1;
	setp.eq.s16	%p8, %rs2, 0;
	@%p8 bra 	BB5_9;

	rem.s32 	%r85, %r15, %r55;
	add.s32 	%r86, %r85, %r55;
	rem.s32 	%r166, %r86, %r55;
	bra.uni 	BB5_10;

BB5_9:
	add.s32 	%r87, %r55, -1;
	min.s32 	%r166, %r15, %r87;

BB5_10:
	add.s32 	%r88, %r166, %r5;
	cvt.s64.s32	%rd13, %r88;
	add.s64 	%rd14, %rd2, %rd13;
	ld.global.nc.u8 	%rs9, [%rd14];
	and.b16  	%rs10, %rs9, 255;
	setp.gt.u16	%p9, %rs10, %rs8;
	cvt.u32.u16	%r89, %rs9;
	and.b32  	%r19, %r89, 255;
	@%p9 bra 	BB5_12;

	add.s32 	%r90, %r6, 1;
	mul.lo.s32 	%r91, %r90, %r6;
	shr.u32 	%r92, %r91, 31;
	add.s32 	%r93, %r91, %r92;
	shr.s32 	%r94, %r93, 1;
	add.s32 	%r167, %r94, %r19;
	bra.uni 	BB5_13;

BB5_12:
	add.s32 	%r95, %r19, 1;
	mul.lo.s32 	%r96, %r95, %r19;
	shr.u32 	%r97, %r96, 31;
	add.s32 	%r98, %r96, %r97;
	shr.s32 	%r99, %r98, 1;
	add.s32 	%r167, %r99, %r6;

BB5_13:
	mul.wide.s32 	%rd15, %r167, 4;
	add.s64 	%rd16, %rd1, %rd15;
	ld.global.nc.f32 	%f9, [%rd16];
	add.f32 	%f2, %f1, %f9;
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16	%p10, %rs3, 0;
	add.s32 	%r23, %r2, -1;
	@%p10 bra 	BB5_15;

	rem.s32 	%r100, %r23, %r56;
	add.s32 	%r101, %r100, %r56;
	rem.s32 	%r168, %r101, %r56;
	bra.uni 	BB5_16;

BB5_15:
	mov.u32 	%r102, 0;
	max.s32 	%r168, %r23, %r102;

BB5_16:
	add.s32 	%r103, %r168, %r4;
	mad.lo.s32 	%r104, %r103, %r55, %r1;
	cvt.s64.s32	%rd17, %r104;
	add.s64 	%rd18, %rd2, %rd17;
	ld.global.nc.u8 	%rs12, [%rd18];
	and.b16  	%rs13, %rs12, 255;
	setp.gt.u16	%p11, %rs13, %rs8;
	cvt.u32.u16	%r105, %rs12;
	and.b32  	%r27, %r105, 255;
	@%p11 bra 	BB5_18;

	add.s32 	%r106, %r6, 1;
	mul.lo.s32 	%r107, %r106, %r6;
	shr.u32 	%r108, %r107, 31;
	add.s32 	%r109, %r107, %r108;
	shr.s32 	%r110, %r109, 1;
	add.s32 	%r169, %r110, %r27;
	bra.uni 	BB5_19;

BB5_18:
	add.s32 	%r111, %r27, 1;
	mul.lo.s32 	%r112, %r111, %r27;
	shr.u32 	%r113, %r112, 31;
	add.s32 	%r114, %r112, %r113;
	shr.s32 	%r115, %r114, 1;
	add.s32 	%r169, %r115, %r6;

BB5_19:
	mul.wide.s32 	%rd19, %r169, 4;
	add.s64 	%rd20, %rd1, %rd19;
	ld.global.nc.f32 	%f10, [%rd20];
	add.f32 	%f3, %f2, %f10;
	add.s32 	%r31, %r2, 1;
	and.b16  	%rs15, %rs3, 255;
	setp.eq.s16	%p12, %rs15, 0;
	@%p12 bra 	BB5_21;

	rem.s32 	%r116, %r31, %r56;
	add.s32 	%r117, %r116, %r56;
	rem.s32 	%r170, %r117, %r56;
	bra.uni 	BB5_22;

BB5_21:
	add.s32 	%r118, %r56, -1;
	min.s32 	%r170, %r31, %r118;

BB5_22:
	add.s32 	%r119, %r170, %r4;
	mad.lo.s32 	%r120, %r119, %r55, %r1;
	cvt.s64.s32	%rd21, %r120;
	add.s64 	%rd22, %rd2, %rd21;
	ld.global.nc.u8 	%rs16, [%rd22];
	and.b16  	%rs17, %rs16, 255;
	setp.gt.u16	%p13, %rs17, %rs8;
	cvt.u32.u16	%r121, %rs16;
	and.b32  	%r35, %r121, 255;
	@%p13 bra 	BB5_24;

	add.s32 	%r122, %r6, 1;
	mul.lo.s32 	%r123, %r122, %r6;
	shr.u32 	%r124, %r123, 31;
	add.s32 	%r125, %r123, %r124;
	shr.s32 	%r126, %r125, 1;
	add.s32 	%r171, %r126, %r35;
	bra.uni 	BB5_25;

BB5_24:
	add.s32 	%r127, %r35, 1;
	mul.lo.s32 	%r128, %r127, %r35;
	shr.u32 	%r129, %r128, 31;
	add.s32 	%r130, %r128, %r129;
	shr.s32 	%r131, %r130, 1;
	add.s32 	%r171, %r131, %r6;

BB5_25:
	mul.wide.s32 	%rd23, %r171, 4;
	add.s64 	%rd24, %rd1, %rd23;
	ld.global.nc.f32 	%f11, [%rd24];
	add.f32 	%f14, %f3, %f11;
	setp.eq.s32	%p14, %r57, 1;
	@%p14 bra 	BB5_39;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16	%p15, %rs4, 0;
	add.s32 	%r39, %r3, -1;
	@%p15 bra 	BB5_28;

	rem.s32 	%r132, %r39, %r57;
	add.s32 	%r133, %r132, %r57;
	rem.s32 	%r172, %r133, %r57;
	bra.uni 	BB5_29;

BB5_28:
	mov.u32 	%r134, 0;
	max.s32 	%r172, %r39, %r134;

BB5_29:
	mad.lo.s32 	%r135, %r172, %r56, %r2;
	mad.lo.s32 	%r136, %r135, %r55, %r1;
	cvt.s64.s32	%rd25, %r136;
	add.s64 	%rd26, %rd2, %rd25;
	ld.global.nc.u8 	%rs19, [%rd26];
	and.b16  	%rs20, %rs19, 255;
	setp.gt.u16	%p16, %rs20, %rs8;
	cvt.u32.u16	%r137, %rs19;
	and.b32  	%r43, %r137, 255;
	@%p16 bra 	BB5_31;

	add.s32 	%r138, %r6, 1;
	mul.lo.s32 	%r139, %r138, %r6;
	shr.u32 	%r140, %r139, 31;
	add.s32 	%r141, %r139, %r140;
	shr.s32 	%r142, %r141, 1;
	add.s32 	%r173, %r142, %r43;
	bra.uni 	BB5_32;

BB5_31:
	add.s32 	%r143, %r43, 1;
	mul.lo.s32 	%r144, %r143, %r43;
	shr.u32 	%r145, %r144, 31;
	add.s32 	%r146, %r144, %r145;
	shr.s32 	%r147, %r146, 1;
	add.s32 	%r173, %r147, %r6;

BB5_32:
	mul.wide.s32 	%rd27, %r173, 4;
	add.s64 	%rd28, %rd1, %rd27;
	ld.global.nc.f32 	%f12, [%rd28];
	add.f32 	%f5, %f14, %f12;
	add.s32 	%r47, %r3, 1;
	and.b16  	%rs22, %rs4, 255;
	setp.eq.s16	%p17, %rs22, 0;
	@%p17 bra 	BB5_34;

	rem.s32 	%r148, %r47, %r57;
	add.s32 	%r149, %r148, %r57;
	rem.s32 	%r174, %r149, %r57;
	bra.uni 	BB5_35;

BB5_34:
	add.s32 	%r150, %r57, -1;
	min.s32 	%r174, %r47, %r150;

BB5_35:
	mad.lo.s32 	%r151, %r174, %r56, %r2;
	mad.lo.s32 	%r152, %r151, %r55, %r1;
	cvt.s64.s32	%rd29, %r152;
	add.s64 	%rd30, %rd2, %rd29;
	ld.global.nc.u8 	%rs23, [%rd30];
	and.b16  	%rs24, %rs23, 255;
	setp.gt.u16	%p18, %rs24, %rs8;
	cvt.u32.u16	%r153, %rs23;
	and.b32  	%r51, %r153, 255;
	@%p18 bra 	BB5_37;

	add.s32 	%r154, %r6, 1;
	mul.lo.s32 	%r155, %r154, %r6;
	shr.u32 	%r156, %r155, 31;
	add.s32 	%r157, %r155, %r156;
	shr.s32 	%r158, %r157, 1;
	add.s32 	%r175, %r158, %r51;
	bra.uni 	BB5_38;

BB5_37:
	add.s32 	%r159, %r51, 1;
	mul.lo.s32 	%r160, %r159, %r51;
	shr.u32 	%r161, %r160, 31;
	add.s32 	%r162, %r160, %r161;
	shr.s32 	%r163, %r162, 1;
	add.s32 	%r175, %r163, %r6;

BB5_38:
	mul.wide.s32 	%rd31, %r175, 4;
	add.s64 	%rd32, %rd1, %rd31;
	ld.global.nc.f32 	%f13, [%rd32];
	add.f32 	%f14, %f5, %f13;

BB5_39:
	shl.b64 	%rd33, %rd4, 2;
	add.s64 	%rd34, %rd3, %rd33;
	st.global.f32 	[%rd34], %f14;

BB5_40:
	ret;
}


`
)
