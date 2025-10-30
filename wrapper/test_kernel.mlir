module {
  func.func @kernel_call(%arg0: !torch.vtensor<[1,64],f32>, %arg1: !torch.vtensor<[64,100],f32>) -> !torch.vtensor<[1,100],f32> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1,64],f32>, !torch.vtensor<[64,100],f32> -> !torch.vtensor<[1,100],f32>
    return %0 : !torch.vtensor<[1,100],f32>
  }
}
