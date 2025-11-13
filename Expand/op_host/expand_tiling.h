
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ExpandTilingData)
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, outer);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, repeater);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, inner);
  TILING_DATA_FIELD_DEF(int32_t,size);
  TILING_DATA_FIELD_DEF(int32_t,outputsize);
  TILING_DATA_FIELD_DEF(int32_t,Expandsize);
  TILING_DATA_FIELD_DEF(int32_t,datatypesize);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Expand, ExpandTilingData)
}
