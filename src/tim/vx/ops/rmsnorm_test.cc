/****************************************************************************
 *
 *    Copyright (c) 2020-2023 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include "gtest/gtest.h"
#include "test_utils.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/rmsnorm.h"

#ifdef VSI_FEAT_OP_RMSNORM
TEST(RMSNorm, shape_4_2_fp32_axis_0) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({4, 2});
  tim::vx::ShapeType scale_shape({4});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec scale_spec(tim::vx::DataType::FLOAT32, scale_shape,
                                 tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto scale_tensor = graph->CreateTensor(scale_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1,-1,1,-1,
      -2,2,-2,2
  };
  std::vector<float> scale_data = { 1,1,1,1 };
  std::vector<float> golden = {
      1,-1,1,-1,
      -2,2,-2,2
  };

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));
  EXPECT_TRUE(scale_tensor->CopyDataToTensor(scale_data.data(),
                                             scale_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::RMSNorm>(0,1e-6);
  (*op).BindInput(input_tensor).BindInput(scale_tensor).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output,1e-5f));
}
#endif