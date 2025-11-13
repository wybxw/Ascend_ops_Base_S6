
#include "expand_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ExpandTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);

  int32_t data_sz = 1;
  int dim = x1_shape->GetStorageShape().GetDimNum();
  const long int* y_dim = context->GetAttrs()->GetListInt(0)->GetData();
  int32_t outer[3]={0};
  int32_t inner[3]={0};
  int32_t repeater[3]={0};
  int64_t output_size = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i),
    output_size *= y_dim[i];
  tiling.set_size(data_sz);
  context->SetBlockDim(1);
  int out=data_sz;
  int in=1;
  int repeat=1;
  int i=0;
  int j=0;
  while(dim-i-1>=0){
    
        
    if(x1_shape->GetStorageShape().GetDim(dim-1-i)==1){
        repeat*=y_dim[dim-1-i];
    }else{
        if(repeat!=1){
            outer[j]=out;
            inner[j]=in;
            repeater[j]=repeat;
            in*=repeater[j];
            j++;
        }
        repeat=1;
        out/=x1_shape->GetStorageShape().GetDim(dim-1-i);
        in*=x1_shape->GetStorageShape().GetDim(dim-1-i);
    }
    ++i;
  }
  if(repeat!=1){
      outer[j]=out;
      inner[j]=in;
      repeater[j]=repeat;
      j++;
  }
  ge::DataType inputDataType = context->GetInputDesc(0)->GetDataType();
  tiling.set_Expandsize(j);
  tiling.set_outer(outer);
  tiling.set_inner(inner);
  tiling.set_repeater(repeater);
  tiling.set_outputsize(output_size);
  
  uint32_t inputDataTypeSize = 0;
  switch(inputDataType) {
    case ge::DT_INT8:  inputDataTypeSize = 1;break;
    case ge::DT_INT16: inputDataTypeSize = 2;break;
    case ge::DT_INT32: inputDataTypeSize = 4;break;
    case ge::DT_INT64: inputDataTypeSize = 8;break;
    case ge::DT_UINT8:  inputDataTypeSize = 1;break;
    case ge::DT_UINT16: inputDataTypeSize = 2;break;
    case ge::DT_UINT32: inputDataTypeSize = 4;break;
    case ge::DT_UINT64: inputDataTypeSize = 8;break;
    case ge::DT_FLOAT16: inputDataTypeSize = 2;break;
    case ge::DT_FLOAT: inputDataTypeSize = 4;break;
    case ge::DT_BOOL: inputDataTypeSize = 1;break;
    case ge::DT_BF16: inputDataTypeSize = 2;break;
    default: inputDataTypeSize = 4;break; // 默认 int32
  }
  // if(inputDataTypeSize == 1) {
  //   context->SetTilingKey(1);
  // }else if(inputDataTypeSize == 2) {
  //   context->SetTilingKey(2);
  // }else if(inputDataTypeSize == 4) {
  //   context->SetTilingKey(3);
  // }else if(inputDataTypeSize == 8) {
  //   context->SetTilingKey(4);
  // }
  tiling.set_datatypesize(inputDataTypeSize);
  size_t usrSize = output_size * inputDataTypeSize+1024;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  int32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
  if (j>1)//广播多次时才需要workspace
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const size_t x1_dim = context->GetAttrs()->GetListInt(0)->GetSize();   
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(x1_dim);
    const long int * pt = context->GetAttrs()->GetListInt(0)->GetData();
    for (int i = 0; i < x1_dim ;i++) {
        y_shape->SetDim(i, pt[i]);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Expand : public OpDef {
public:
    explicit Expand(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // this->Input("x")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // this->Output("out")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        this->Attr("size").ListInt();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Expand);
}
