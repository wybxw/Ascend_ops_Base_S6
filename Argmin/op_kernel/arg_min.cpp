

#include "kernel_operator.h"
#include <limits>
#include <type_traits>
#include <utility>
using namespace AscendC;

/* 数值比较类型映射：bfloat16 在向量比较/计算阶段提升到 float */
template <typename T>
struct CmpType { using type = T; };
template <> struct CmpType<bfloat16_t> { using type = float; };

/* 支持类型集合 */
template <typename T> struct IsArgMinSupported : std::false_type {};
template <> struct IsArgMinSupported<half>       : std::true_type {};
template <> struct IsArgMinSupported<bfloat16_t> : std::true_type {};
template <> struct IsArgMinSupported<float>      : std::true_type {};
template <> struct IsArgMinSupported<int8_t>     : std::true_type {};
template <> struct IsArgMinSupported<uint8_t>    : std::true_type {};
template <> struct IsArgMinSupported<int16_t>    : std::true_type {};
template <> struct IsArgMinSupported<int32_t>    : std::true_type {};
template <> struct IsArgMinSupported<int64_t>    : std::true_type {};

template <typename T>
class KernelArgMin
{
public:
    using ValueT = T;
    using IndexT = int64_t;
    using CmpT   = typename CmpType<ValueT>::type;
    static_assert(IsArgMinSupported<T>::value,
                  "KernelArgMin: only float/half/bfloat16/int8/uint8/int16/int32/int64");

    static constexpr int32_t ALIGNED = 32 / sizeof(ValueT);

    // 与 CmpT 等长的无符号整型，用于 reinterpret 索引
    using IdxIntT =
        std::conditional_t<sizeof(CmpT) == 1, uint8_t,
        std::conditional_t<sizeof(CmpT) == 2, uint16_t,
        std::conditional_t<sizeof(CmpT) == 4, uint32_t,
        std::conditional_t<sizeof(CmpT) == 8, uint64_t, void>>>>;

    static constexpr CmpT CmpT_MAX =
        (std::numeric_limits<CmpT>::has_infinity
             ? std::numeric_limits<CmpT>::infinity()
             : std::numeric_limits<CmpT>::max());

    __aicore__ KernelArgMin() = default;

    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR out_idx_gm, GM_ADDR,
                                const ArgMinTilingData &t, TPipe *pipe_ptr)
    {
        blockIdx  = GetBlockIdx();
        blockNum  = GetBlockNum();
        inner     = t.inner;
        outer     = t.outer;
        totalSize = t.size;
        stride_m  = t.stride_m;
        stride    = inner * stride_m;
        planes    = outer / stride_m;
        pipe      = pipe_ptr;
        inner_last= t.inner - 1;
        xxGm.SetGlobalBuffer(reinterpret_cast<__gm__ ValueT *>(x_gm), totalSize);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ IndexT*>(out_idx_gm), outer);

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(x_gm), totalSize);
        DataCachePreload(xGm, int64_t(0));

        if (TILING_KEY_IS(1)) {
            // Slice 单流水：深度=1
            pipe->InitBuffer(inSliceQueue, 1, (TILE_INNER) * sizeof(ValueT) + 32);
            pipe->InitBuffer(bufMinIdx,    32);
        } else if (TILING_KEY_IS(0)) {
            // 平面路径：行读取双缓冲；写回用 VECOUT 双缓冲
            pipe->InitBuffer(rowQueue,     2, (TILE_COL) * sizeof(ValueT)    + 32);
            pipe->InitBuffer(bufcmpMask,      (TILE_COL + 7) / 8           + 32);
            pipe->InitBuffer(bufCastIdx,      (TILE_COL) * sizeof(int32_t)   + 32);
            pipe->InitBuffer(bufRow,          (TILE_COL) * sizeof(CmpT) + 32);
            pipe->InitBuffer(outIdxQueue,  1, (TILE_COL) * sizeof(IndexT) + 256);
            
        }
    }

    __aicore__ inline void Process()
    {

        
        if (TILING_KEY_IS(1)) {
            for (uint32_t s = blockIdx; s < outer; s += blockNum) {
                ReduceContiguousSlice(s);
            }
        } else if (TILING_KEY_IS(0)) {
            for (uint32_t p = blockIdx; p < planes; p += blockNum) {
                ReducePlane(p);
            }
        }
    }

private:
    /* -------- 工具函数 -------- */
    static constexpr uint32_t VEC_BYTES = 256;

    __aicore__ static inline uint32_t Align32Elems(uint32_t len)
    {
        return (len + ALIGNED - 1) & ~(ALIGNED - 1);
    }
    __aicore__ static inline uint32_t VecElems() { return VEC_BYTES / sizeof(CmpT); }
    __aicore__ static inline uint32_t RoundUpTo(uint32_t n, uint32_t a) { return (n + a - 1) / a * a; }
    __aicore__ static inline uint32_t CmpAlignedLen(uint32_t len) { return RoundUpTo(len, VecElems()); }
    __aicore__ static inline uint32_t Align8Elems(uint32_t len) { return RoundUpTo(len, 8); }

    /* --- 连续维: copyin / compute / copyout --- */
    __aicore__ inline void SliceCopyIn(uint32_t gmPos, uint32_t validLen)
    {
        auto t = inSliceQueue.AllocTensor<ValueT>();
        uint32_t alignedLen = Align32Elems(validLen);
        DataCopy(t, xxGm[gmPos], alignedLen);
        inSliceQueue.EnQue(t);
    }

    __aicore__ inline void SliceCompute(uint32_t baseOffset,
                                        uint32_t validLen,
                                        CmpT &gMin,
                                        IndexT &gIdx)
    {
        auto tile    = inSliceQueue.DeQue<ValueT>();
        auto answer  = bufMinIdx.Get<ValueT>();

        if constexpr (std::is_same<ValueT, half>::value ||
                      std::is_same<ValueT, float>::value)
        {
            ReduceMin(answer, tile, tile, validLen, true);
            if (static_cast<float>(answer.GetValue(0)) < static_cast<float>(gMin)) {
                gMin = answer.GetValue(0);
                CmpT idx = answer.GetValue(1);
                gIdx = baseOffset + *reinterpret_cast<IdxIntT*>(&idx);
            }
        }
        else if constexpr (std::is_same<ValueT, bfloat16_t>::value)
        {
            // 保持现状（按需求未实现该路径）
        }
        else
        {
            for (uint32_t i = 0; i < validLen; ++i) {
                if (tile.GetValue(i) < gMin) {
                    gMin = tile.GetValue(i);
                    gIdx = baseOffset + i;
                }
            }
        }
    }

    __aicore__ inline void SliceCopyOut(uint32_t outPos, IndexT idxVal)
    {
        outGm(outPos) = idxVal;
    }

    __aicore__ inline void ReduceContiguousSlice(uint32_t slice)
    {
        uint32_t base = slice * inner;
        CmpT    gMin = CmpT_MAX;
        IndexT  gIdx = 0;

        uint32_t chunk;
        for (uint32_t done = 0; done < inner; done += chunk) {
            chunk = min(TILE_INNER, inner - done);
            SliceCopyIn(base + done, chunk);
            SliceCompute(done, chunk, gMin, gIdx);
        }
        SliceCopyOut(slice, gIdx);
    }



    /* --- Plane: 行 copyin / 行 compute / 块 copyout（双缓冲流水） --- */
    __aicore__ inline void PlaneRowCopyIn(const uint32_t &gmRowPos, const uint32_t &validLen)
    {
        auto row = rowQueue.AllocTensor<ValueT>();
        uint32_t alignedLen = Align32Elems(validLen);
        DataCopy(row, xxGm[gmRowPos], alignedLen);       // 提交 MTE2
        rowQueue.EnQue(row);
    }

    __aicore__ inline void PlaneInit(LocalTensor<CmpT>   &minVals,
                                     LocalTensor<float> &CastIdx,
                                     const uint32_t &len)
    {
        auto row = rowQueue.DeQue<ValueT>();
        if constexpr (std::is_same<ValueT, bfloat16_t>::value) {
            Cast(minVals,row,AscendC::RoundMode::CAST_NONE,len);//需要cast
        }else {
            DataCopy(minVals,row,len+32);//直接拷贝
        }
        // 初始化索引缓存为 0
        AscendC::Duplicate(CastIdx.ReinterpretCast<int32_t>(),0,len);
        rowQueue.FreeTensor(row);
    }

    __aicore__ inline void PlaneComputeRow(LocalTensor<CmpT> &minVals,
                                           LocalTensor<float> &CastIdx,
                                           LocalTensor<CmpT> &rowbuf,
                                           const uint32_t &len,
                                           const float &r)//r实际上是整数
    {
        LocalTensor<ValueT> row = rowQueue.DeQue<ValueT>();
        LocalTensor<CmpT> bufrow;
        const uint32_t cmpLen = CmpAlignedLen(static_cast<uint32_t>(len));
        auto cmpMask = bufcmpMask.Get<uint8_t>();
        if constexpr (std::is_same<ValueT, bfloat16_t>::value) {
            bufrow = bufRow.Get<CmpT>();
            Cast(bufrow,row,AscendC::RoundMode::CAST_NONE,len);//把row cast 到bufrow
        }else bufrow = row;//否则直接引用row

        if constexpr (std::is_same<CmpT, half>::value ||
                      std::is_same<CmpT, float>::value)
        {
            Compare(cmpMask, bufrow, minVals, AscendC::CMPMODE::GE, cmpLen);
            Select(CastIdx, cmpMask, CastIdx, r,AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Min(minVals, bufrow, minVals, len);
        }
        else if constexpr (std::is_same<CmpT, int32_t>::value)
        {
            Sub(rowbuf, bufrow, minVals, len);
            CompareScalar(cmpMask, rowbuf, 0, AscendC::CMPMODE::GE, cmpLen);
            Select(CastIdx, cmpMask, CastIdx, (r),AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Min(minVals, bufrow, minVals, len);
        }
        rowQueue.FreeTensor(row);
    }
    __aicore__ inline void ReducePlane(const uint32_t &plane)
    {
        const uint32_t planeBase = plane * stride;
        uint32_t chunk;
        
        
        auto CastIdx = bufCastIdx.Get<float>();
        uint32_t off = 0;
        // for (; off < stride_m; off += chunk)//因为TILE_COL提升到了10240 以及大于单维最大值10000 可以不用循环
        // {
            
        chunk = min(TILE_COL, stride_m);
        
        PlaneRowCopyIn(planeBase, chunk);
        auto minIdx  = outIdxQueue.AllocTensor<IndexT>();
        auto minVals = minIdx.ReinterpretCast<CmpT>();//因为minIdx只会在最后cast时用到,先把他当minVals复用
        auto rowbuf = minVals[TILE_COL + 32];

        PlaneInit(minVals,CastIdx,chunk);

        if (inner > 1) {
            PlaneRowCopyIn(planeBase + 1 * stride_m, chunk);
        }
        for (uint32_t r = 1; r + 1 < inner; ++r)
        {
            PlaneRowCopyIn(planeBase + (r + 1) * stride_m, chunk);
            PlaneComputeRow(minVals,CastIdx,rowbuf,chunk, *reinterpret_cast<float*>(&r));
        }

        if (inner > 1) {
            PlaneComputeRow(minVals,CastIdx,rowbuf,chunk, *reinterpret_cast<float*>(&inner_last));
        }
        
        Cast(minIdx, CastIdx.ReinterpretCast<int32_t>(),AscendC::RoundMode::CAST_NONE,chunk);
        outIdxQueue.EnQue(minIdx);
        outIdxQueue.DeQue<IndexT>();
        DataCopy(outGm[plane * stride_m], minIdx, chunk+32);
        outIdxQueue.FreeTensor(minIdx);
        // }
    }

private:
    static constexpr uint32_t TILE_INNER = std::is_same<ValueT, int64_t>::value ? 12288 : 24576;//310B上UB大小248K 如果在其他型号上跑可以适当调大/调小
    static constexpr uint32_t TILE_COL   = std::is_same<ValueT, int64_t>::value ? 5120  : 10240;//310B上UB大小248K 如果在其他型号上跑可以适当调大/调小

    GlobalTensor<uint64_t>   xGm;
    GlobalTensor<ValueT> xxGm;
    GlobalTensor<IndexT> outGm;

    TPipe *pipe;

    // Slice: 深度1；Plane: 深度2
    TQue<TPosition::VECIN,  1> inSliceQueue;
    TQue<TPosition::VECOUT, 1> outIdxQueue; // plane 用
    TQue<TPosition::VECIN,  2> rowQueue;
    TBuf<TPosition::VECCALC> bufRow;
    TBuf<TPosition::VECCALC> bufCastVals;
    TBuf<TPosition::VECCALC> bufMinIdx;  // Slice ReduceMin 使用
    TBuf<TPosition::VECCALC> bufcmpMask;
    TBuf<TPosition::VECCALC> bufCastIdx;

    // AscendC::TQueSync<PIPE_V,   PIPE_MTE3> sync_V_to_MTE3;
    // AscendC::TQueSync<PIPE_MTE2, PIPE_V>   sync_MTE2_to_V;
    uint32_t inner_last;
    uint32_t inner, outer, totalSize, stride_m, stride;
    uint32_t blockIdx, blockNum;
    uint32_t planes;
};

extern "C" __global__ __aicore__ void arg_min(GM_ADDR x, GM_ADDR out_idx,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelArgMin<DTYPE_X> op;
    TPipe pipe;
    op.Init(x, out_idx, workspace, tilingData, &pipe);
    op.Process();
}