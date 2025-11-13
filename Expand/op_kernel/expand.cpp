// ...existing code...
#include "kernel_operator.h"
#include <type_traits>
// #include <iostream>
// #include <algorithm>
// using namespace std;
using namespace AscendC;

// UB 宏
static constexpr int32_t UB_BYTES = 240* 1024;       // 248KB
static constexpr int32_t UB_BUF_RESERVE = 0 * 1024; // 4KB 保留给控制结构等
static constexpr int32_t MIN_BLOCK_BYTES = 32;        // 32B 对齐最小块
static constexpr int32_t BUFFER_NUM = 2;
template <typename T>
__aicore__ inline T min(T a, T b)
{
    return a < b ? a : b;
}
template <typename T>
__aicore__ inline T max(T a, T b)
{
    return a > b ? a : b;
}
__aicore__ inline int32_t gcd_32(int32_t a, int32_t b) {
    if (a == 0) return b;
    if (b == 0) return a;
    while (b) {
        uint64_t r = a % b;
        a = b; b = r;
    }
    return a;
}
template<std::size_t Bytes> struct int_of_bytes;
template<> struct int_of_bytes<1> { using type = std::int8_t;  };
template<> struct int_of_bytes<2> { using type = std::int16_t; };
template<> struct int_of_bytes<4> { using type = std::int32_t; };
template<> struct int_of_bytes<8> { using type = std::int64_t; };

template<std::size_t Bytes>
using int_of_bytes_t = typename int_of_bytes<Bytes>::type;

template <typename T>
constexpr bool KEXP_IsAllowedType_v = std::disjunction_v<
    std::is_same<T, int8_t>,
    std::is_same<T, int16_t>,
    std::is_same<T, int32_t>,
    std::is_same<T, int64_t>
    >;
template <typename T, typename = std::enable_if_t<KEXP_IsAllowedType_v<T>>>
class KernelExpand
{
    static_assert(KEXP_IsAllowedType_v<T>, "");

public:
    __aicore__ inline KernelExpand() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, ExpandTilingData &tiling, AscendC::TPipe *pipein)
    {
        this->pipe = pipein;
        this->blockIdx = AscendC::GetBlockIdx();
        this->blockStride = AscendC::GetBlockNum();

        this->num_tile_entries = tiling.Expandsize;
        for (int i = 0; i < 3; ++i)
        {
            this->outer[i] = tiling.outer[i];
            this->repeater[i] = tiling.repeater[i];
            this->inner[i] = tiling.inner[i];
            // printf("outer:%d repeat:%d inner:%d\n",this->outer[i],this->repeater[i],this->inner[i]);
        }
        this->tiling_size = tiling.size;
        this->outputsize = tiling.outputsize;
        this->total_outer_repeats = tiling.Expandsize;
        this->ws_addr = workspace;
        this->src_addr = src;
        this->dst_addr = dst;
        this->dtype_bytes = sizeof(T);
        int32_t per_buf_bytes =  UB_BYTES/BUFFER_NUM;                 // 两个 vec 缓冲（逻辑上） 84KB
        this->ub_buf_elems = per_buf_bytes / this->dtype_bytes;
        this->align_elems = ( 32 / this->dtype_bytes );
        // 初始化队列并一次性分配 vec 缓冲（现实可能只分配一个并复用）
        pipe->InitBuffer(Queue, BUFFER_NUM, per_buf_bytes);
    }
    __aicore__ inline void Process()
    {
        for (int32_t idx = 0; idx < this->total_outer_repeats; ++idx)
        {

            if ((this->total_outer_repeats & 1) == ((idx + 1) & 1))
            {
                srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ws_addr));
                src32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ws_addr));
                dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_addr));
                dst32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dst_addr));
            }
            else
            {
                srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_addr));
                src32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dst_addr));
                dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ws_addr));
                dst32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ws_addr));
            }
            if (idx == 0)
            {
                srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src_addr));
                src32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(src_addr));
            }
            int32_t step = 1;
            if(inner[idx]*sizeof(T)>=32)//inner中等大小
                step = max(1,min(outer[idx],ub_buf_elems / inner[idx]));
            for (int32_t outer_idx = step*this->blockIdx; outer_idx < this->outer[idx]; outer_idx+=step*this->blockStride)
            {
                performTileBroadcast(idx, outer_idx,min(step,outer[idx]-outer_idx));
            }
        }
    }
    // 计算基址（以元素计）
    __aicore__ inline int32_t compute_in_base(int tile_idx, int32_t outer_index)
    {
        return outer_index * (int32_t)this->inner[tile_idx];
    }
    __aicore__ inline int32_t compute_out_base(int tile_idx, int32_t outer_index)
    {
        return outer_index * (int32_t)(this->inner[tile_idx] * this->repeater[tile_idx]);
    }

    __aicore__ inline bool is32AlignedElem(int32_t offset_elems) const
    {
        return (offset_elems * this->dtype_bytes) % 32 == 0;
    }

    // 对齐拷贝
    __aicore__ inline void MyCopy(AscendC::LocalTensor<T> &dstBase,
                                   int32_t elems)
    {
        //当行大小中等偏上时可以直接pad 拷贝，然后非对齐搬出
        //但当行大小 太小时，pad区域太多。应该考虑 标量补齐32B对齐
        elems = (elems+this->align_elems-1)/this->align_elems*this->align_elems;//pad到对齐32B
        if constexpr (std::is_same_v<T, int8_t>) Adds<int16_t>(dstBase[elems].template ReinterpretCast<int16_t>(),dstBase.template ReinterpretCast<int16_t>(),0,elems>>1);
        else if constexpr(std::is_same_v<T, int16_t>) Adds<int16_t>(dstBase[elems],dstBase,0,elems);    
        else if constexpr(std::is_same_v<T, int32_t>) Adds<int32_t>(dstBase[elems],dstBase,0,elems);    
        else if constexpr (std::is_same_v<T, int64_t>) Adds<int32_t>(dstBase[elems].template ReinterpretCast<int32_t>(),dstBase.template ReinterpretCast<int32_t>(),0,elems<<1);

    }
    __aicore__ inline void MyFillPad(AscendC::LocalTensor<T> &dstBase,int32_t &filled)
    {
        //使用标量 将其复制到一个可接受的大小
        int repeat = this->align_elems/gcd_32(filled,this->align_elems);
        if(repeat*filled*sizeof(T) <= 4 * 1024){
            for(int i=1;i<repeat;i++)
                for(int j=0;j<filled;j++)
                    dstBase.SetValue(i*filled + j,dstBase.GetValue(j));
            filled *= repeat;
        }
        
    }
    __aicore__ inline void copyIn(
        int32_t in_base_elem,
        int32_t offset_elem,
        int32_t cur_elems)
    {
        auto vecbuf = Queue.AllocTensor<T>();
        AscendC::DataCopy(vecbuf, srcGm[in_base_elem + offset_elem], cur_elems + align_elems - 1);
        Queue.EnQue<T>(vecbuf);
    }
    __aicore__ inline int32_t computeExpandVec(int32_t cur_elems,int32_t inner_elems,int32_t repeat)
    {
        int32_t filled = cur_elems ;
        auto vecbuf = Queue.DeQue<T>();
        if (inner_elems == 1)
        {
            
            filled = min(this->ub_buf_elems, repeat); // 最多repeat成一行
            // printf("filled:%d\n",filled);
            T value = vecbuf.GetValue(0);
            if constexpr (std::is_same_v<T, int8_t>){
                // 按位解释为无符号
                Duplicate<int16_t>(vecbuf.template ReinterpretCast<int16_t>(), static_cast<std::int16_t>(static_cast<std::uint8_t>(value) * 0x0101u), (filled) >> 1);
                if(filled&1) vecbuf.SetValue(filled-1,value);
            }
            else if constexpr (std::is_same_v<T, int16_t>||std::is_same_v<T, int32_t>) Duplicate<T>(vecbuf, value, filled);
            else if constexpr (std::is_same_v<T, int64_t>){
                uint8_t tmp = (filled+31)/32*32;
                Duplicate<int32_t>(vecbuf.template ReinterpretCast<int32_t>(), static_cast<int32_t>((value >> 32) & 0xffffffff), maskhigh, tmp, 1, 8);
                Duplicate<int32_t>(vecbuf.template ReinterpretCast<int32_t>(), static_cast<int32_t>(value & 0xffffffff), masklow, tmp, 1, 8);
            }
        }
        else if(inner_elems*sizeof(T)<=32)
        {
            
            //只考虑特别小的case 
            int32_t all =min(ub_buf_elems,repeat*inner_elems);
            if(cur_elems%this->align_elems != 0){
                MyFillPad(vecbuf,filled);
            }
            if(filled % align_elems == 0)
            while (filled * 2 <= all)//倍增不超界
            {
                MyCopy(vecbuf, filled);
                filled <<= 1;//倍增拷贝
            }
        }
        Queue.EnQue<T>(vecbuf);
        return filled;
    }

    // 修正后的 copyOut：多行/单行模式区分
    __aicore__ inline void copyOut(int32_t out_base_elem,
                                   int32_t offset_elems,     // 行内偏移
                                   int32_t inner_elems,     // 一行的元素数 (in)
                                   int32_t fill_elems,      // UB 中有效元素数
                                   int32_t repeat,          // 需要的总行数
                                   int32_t cur_chunk_elems) // 本 chunk 的列数 (cur)
    {
        int32_t rows_in_vec = max(1,fill_elems / inner_elems); // 可以保证fill是inner的整倍数
        auto vecbuf = Queue.DeQue<T>();
        int32_t copy_width = min (cur_chunk_elems,inner_elems);
        int32_t rows_done = 0;
        while (rows_done < repeat)
        {
            int32_t this_rows = min(rows_in_vec, repeat - rows_done);
            int32_t dst_pos = out_base_elem + rows_done * inner_elems + offset_elems;
            MyDataCopyPadOut(vecbuf,dst_pos, this_rows*copy_width);
            rows_done += this_rows;
        }
        Queue.FreeTensor(vecbuf);
    }
    __aicore__ inline void performTileBroadcast(int tile_idx,
                                                int32_t outer_index,int32_t step =1)
    {
        int32_t inner_elems = this->inner[tile_idx]; // in
        int32_t repeat = this->repeater[tile_idx];   // repeat
        int32_t in_base = compute_in_base(tile_idx, outer_index);
        int32_t out_base = compute_out_base(tile_idx, outer_index);

        int32_t chunk_elems = min(this->ub_buf_elems, inner_elems);
        int32_t num_chunks = (inner_elems + chunk_elems - 1) / chunk_elems;
        if(step>1){
            DataCopyExtParams cp_in{static_cast<uint16_t>(step),static_cast<uint32_t>(inner_elems*sizeof(T)),0,0,0};
            DataCopyExtParams cp_out{static_cast<uint16_t>(step),static_cast<uint32_t>(inner_elems*sizeof(T)),0,static_cast<uint32_t>((repeat-1)*inner_elems*sizeof(T)),0};
            auto buf = Queue.AllocTensor<T>();
            
            if constexpr(sizeof(T)==1||sizeof(T)==2||sizeof(T)==4||sizeof(T)==8)
                DataCopyPad(buf,srcGm[in_base],cp_in,{0,0,0,0});
            Queue.EnQue<T>(buf);
            buf = Queue.DeQue<T>();
            if constexpr(sizeof(T)==1||sizeof(T)==2||sizeof(T)==4||sizeof(T)==8)
                for(int i=0;i<repeat;i++)
                    DataCopyPad(dstGm[out_base+i*inner_elems],buf,cp_out);
            Queue.FreeTensor<T>(buf);
        }else
        for (int32_t c = 0; c < num_chunks; ++c)
        {
            int32_t offset = c * chunk_elems;
            int32_t cur = min(chunk_elems, inner_elems - offset);
            copyIn(in_base, offset, cur);
            int32_t fill_elems = computeExpandVec(cur, inner_elems, repeat);
            copyOut(out_base, offset, inner_elems, fill_elems, repeat, cur);
        }
    }
    __aicore__ inline void MyDataCopyPadOut(LocalTensor<T> &vecbuf,int32_t dst_offset, int32_t elems)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same_v<T, int8_t>||std::is_same_v<T, int16_t>||std::is_same_v<T, int32_t>) DataCopyPad<T>(dstGm[dst_offset], vecbuf,copyParams); // copyParams);
        else if constexpr (std::is_same_v<T, int64_t>){
            DataCopyPad<int32_t>(dst32Gm[dst_offset*2], vecbuf.template ReinterpretCast<int32_t>(), copyParams);
        }
    }


private:
    AscendC::GlobalTensor<T> srcGm;
    AscendC::GlobalTensor<T> dstGm;
    AscendC::GlobalTensor<T> wsGm;
    AscendC::GlobalTensor<int32_t> dst32Gm;
    
    AscendC::GlobalTensor<int32_t> src32Gm;
    
    AscendC::TPipe *pipe;
    AscendC::TQueBind<AscendC::TPosition::VECIN,AscendC::TPosition::VECOUT,BUFFER_NUM> Queue;
    
    // AscendC::LocalTensor<T> vecbuf;
    GM_ADDR ws_addr;
    GM_ADDR src_addr;
    GM_ADDR dst_addr;

    int32_t blockIdx;
    int32_t blockStride;
    int32_t num_cores;

    int32_t num_tile_entries;
    int32_t outer[3];
    int32_t repeater[3];
    int32_t inner[3];
    int32_t outputsize;
    int32_t tiling_size;
    int32_t total_outer_repeats;
    int32_t dtype_bytes;
    int32_t ub_buf_elems;
    int32_t align_elems;

    uint64_t masklow[1] = {0x5555555555555555};  // 0101...
    uint64_t maskhigh[1] = {0xaaaaaaaaaaaaaaaa}; // 1010...
    uint64_t mask16[2] = {0xffffffffffffffff, 0xffffffffffffffff};
    uint64_t mask32[1] = {0xffffffffffffffff};
    uint64_t mask64[1] = {0xffffffff};
};

extern "C" __global__ __aicore__ void expand(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0); // 增加这一行
    // assert(sizeof(DTYPE_X)==4||sizeof(DTYPE_X)==8);
    KernelExpand<int_of_bytes_t<sizeof(DTYPE_X)>> op;
    op.Init(src, dst, workspace, tilingData, &pipe);
    op.Process();
    
}


