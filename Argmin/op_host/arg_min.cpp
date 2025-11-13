#include "arg_min_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
// ... TilingFunc ...
// (保持 TilingFunc 不变，问题根源在 InferShape 和 OpDef 的协同)
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ArgMinTilingData tiling;
    const gert::StorageShape* in_shape = context->GetInputShape(0);
    const auto &ss = in_shape->GetStorageShape();
    int32_t rank = ss.GetDimNum();

    uint64_t total_elems = 1;
    for (int i = 0; i < rank; ++i) {
        total_elems *= static_cast<uint64_t>(ss.GetDim(i));
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    int16_t dim_attr = *attrs->GetAttrPointer<int>(0);

    bool global_reduce = (dim_attr == 255);
    int16_t dim = -1;

    if (!global_reduce) {
        if (rank == 0) {
            global_reduce = true;
        } else {
            int64_t d = dim_attr;
            if (d < 0) d += rank;
            if (d < 0 || d >= rank) return ge::GRAPH_FAILED;
            dim = static_cast<int16_t>(d);
        }
    }
    
    uint64_t inner = 0;
    uint64_t outer = 0;
    uint64_t stride_m = 1;

    if (global_reduce) {
        inner = static_cast<uint64_t>(total_elems);
        outer = 1;
        dim = 0; 
    } else {
        inner = static_cast<uint64_t>(ss.GetDim(dim));
        outer = total_elems / inner;
        if (dim < rank - 1) {
            stride_m = 1;
            for (int i = dim + 1; i < rank; ++i) {
                stride_m *= static_cast<uint64_t>(ss.GetDim(i));
            }
        }
    }

    uint64_t elem_bytes = 4;
    auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_INT16) elem_bytes = 2;
    else if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) elem_bytes = 1;
    else if (dtype == ge::DT_INT64) elem_bytes = 8;

    tiling.set_dim(dim); 
    tiling.set_rank(static_cast<uint32_t>(rank));
    tiling.set_inner(inner);
    tiling.set_outer(static_cast<uint32_t>(outer));
    tiling.set_size(static_cast<uint32_t>(total_elems));
    tiling.set_elem_bytes(elem_bytes);
    tiling.set_stride_m(stride_m);
    if(stride_m == 1) context->SetTilingKey(1);
    else context->SetTilingKey(0);
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* in_shape = context->GetInputShape(0);
    gert::Shape* out_shape = context->GetOutputShape(0);

    // --- 遵照您的要求替换回这里的代码 ---
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    int64_t dim_attr = *attrs->GetAttrPointer<int>(0);
    bool keepdim = *attrs->GetAttrPointer<int>(1);

    int32_t rank = in_shape->GetDimNum();
    if (rank == 0) {
        if (keepdim) { out_shape->SetDimNum(1); out_shape->SetDim(0, 1); }
        else { out_shape->SetDimNum(0); }
        return GRAPH_SUCCESS;
    }

    if (dim_attr == 255) { // 全局
        if (keepdim) {
            out_shape->SetDimNum(1);
            for (int i = 0; i < 1; ++i) out_shape->SetDim(i, 1);
        } else {
            out_shape->SetDimNum(0);
        }
        return GRAPH_SUCCESS;
    }

    int64_t d = dim_attr;
    if (d < 0) d += rank;
    if (d < 0 || d >= rank) return GRAPH_FAILED;
    int32_t dim = static_cast<int32_t>(d);

    if (keepdim) {
        out_shape->SetDimNum(rank);
        for (int i = 0; i < rank; ++i) {
            out_shape->SetDim(i, (i == dim) ? 1 : in_shape->GetDim(i));
        }
    } else {
        out_shape->SetDimNum(rank > 0 ? rank - 1 : 0);
        int out_i = 0;
        for (int i = 0; i < rank; ++i) {
            if (i == dim) continue;
            out_shape->SetDim(out_i++, in_shape->GetDim(i));
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    return GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class ArgMin : public OpDef {
public:
    explicit ArgMin(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(255);
        this->Attr("keepdim").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ArgMin);
}