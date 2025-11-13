
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgMinTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(int16_t, dim);         // 被归约轴（-1 表示全局 flatten）
  TILING_DATA_FIELD_DEF(uint32_t, rank);       // input rank
  TILING_DATA_FIELD_DEF(uint32_t, inner);      // 被归约维长度
  TILING_DATA_FIELD_DEF(uint32_t, outer);      // 其余维乘积
  TILING_DATA_FIELD_DEF(uint32_t, elem_bytes); // 每个元素字节数
  TILING_DATA_FIELD_DEF(uint32_t, stride_m);   // 非末维归约时为 ∏_{i>dim} N_i，否则为 1
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgMin, ArgMinTilingData)
}