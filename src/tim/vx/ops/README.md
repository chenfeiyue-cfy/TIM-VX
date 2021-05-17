TIM-VX API |Internal Op |Status
:------    |:----- |:------
Add|ADD|Mapped
Multiply|MULTIPLY|Mapped
Conv2d|CONV2D|Mapped
||CONV_RELU|Deprecated
||CONV_RELU_POOL|Deprecated
||FCL|Deprecated
||FCL_RELU|Deprecated
Softmax|SOFTMAX|Mapped
Pool2d|POOL|Mapped
LeakyRelu|LEAKY_RELU|Mapped
||LRN|Deprecated
Concat|CONCAT|Mapped
Split|SPLIT|Mapped
||NOOP|Unmapped
||ROI_POOL|Unmapped
BatchNorm|BATCH_NORM|Mapped
||PROPOSAL|Unmapped
DeConv2d|DECONVOLUTION|Mapped
Reshape|RESHAPE|Mapped
Transpose|PERMUTE|Mapped
Prelu|PRELU|Mapped
||UPSAMPLE|Unmapped
Relu|RELU|Mapped
||RELUN|Deprecated
||LSTM|Unmapped
Reorg|REORG|Mapped
||VARIABLE|Unmapped
L2Normalization|L2_NORMALIZE|Mapped
FullyConnected|FCL2|Mapped
||POOLWITHARGMAX|Unmapped
ArgMax|ARGMAX|Mapped
Maximum|MAXIMUM|Mapped
L2Normalization|L2NORMALIZESCALE|Mapped
||CROP|Unmapped
Sub|SUBTRACT|Mapped
Relu6|RELU6|Mapped
Sigmoid|SIGMOID|Mapped
Tanh|TANH|Mapped
Sqrt|SQRT|Mapped
Rsqrt|RSQRT|Mapped
SoftRelu|SOFTRELU|Unmapped
Div|DIVIDE|Mapped
Dropout|DROPOUT|Mapped
||SHUFFLECHANNEL|Unmapped
Resize|RESIZE|Mapped
Reverse|REVERSE|Mapped
DepthToSpace|DEPTH2SPACE|Mapped
SpaceToDepth|SPACE2DEPTH|Mapped
DataConvert|DATACONVERT|Mapped
||SCALE|Unmapped
Slice|SLICE|Mapped
Elu|ELU|Mapped
Batch2Space|BATCH2SPACE|Mapped
Space2Batch|SPACE2BATCH|Mapped
Pad|PAD|Mapped
||IMAGEPROCESS|Unmapped
||MATRIXMUL|Unmapped
||LSTMUNIT|Unmapped
||LAYER_NORM|Unmapped
ReduceMin|REDUCE_MIN|Mapped
ReduceMax|REDUCE_MAX|Mapped
ReduceAny|REDUCE_ANY|Mapped
ReduceProd|REDUCE_PROD|Mapped
ReduceMean|REDUCE_MEAN|Mapped
||INSTANCE_NORM|Unmapped
||TENSORSTACKCONCAT|Unmapped
StridedSlice|STRIDED_SLICE|Mapped
||SIGNAL_FRAME|Unmapped
||A_TIMES_B_PLUS_C|Unmapped
||SVDF|Unmapped
Abs|ABS|Mapped
||CONV1D|Unmapped
NBG|NBG|Mapped
||CONCATSHIFT|Unmapped
LocalResponseNormalization|LRN2|Mapped
Greater|RELATIONAL_OPS_GREATER|Mapped
GreaterOrEqual|RELATIONAL_OPS_GREATER_EQUAL|Mapped
Less|RELATIONAL_OPS_LESS|Mapped
LessOrEqual|RELATIONAL_OPS_LESS_EQUAL|Mapped
Equal|RELATIONAL_OPS_EQUAL|Mapped
NotEqual|RELATIONAL_OPS_NOT_EQUAL|Mapped
||SYNC_HOST|Unmapped
Pow|POW|Mapped
||FLOORDIV|Unmapped
Minimum|MINIMUM|Mapped
||SPATIAL_TRANSFORMER|Unmapped
And/Or|LOGICAL_OPS|Mapped
Select|SELECT|Mapped
||LSTMUNIT_ACTIVATION|Unmapped
||LSTMUNIT_OVXLIB|Unmapped
||TENSOR_ADD_MEAN_STDDEV_NORM|Unmapped
Relu1|RELU1|Mapped
Stack|STACK|Mapped
Floor|FLOOR|Mapped
Square|SQUARE|Mapped
Neg|NEG|Mapped
Exp|EXP|Mapped
||LSTM_OVXLIB|Unmapped
||PRE_PROCESS_TENSOR|Unmapped
||HASHTABLE_LOOKUP|Unmapped
||EMBEDDING_LOOKUP|Unmapped
||LSH_PROJECTION|Unmapped
||RNN|Unmapped
Clip|CLIP|Mapped
||POST_PROCESS|Unmapped
||PRE_PROCESS_GRAY|Unmapped
||UNSTACK|Unmapped
||PRE_PROCESS_RGB|Unmapped
||PRE_PROCESS|Unmapped
AddN|ADDN|Mapped
||PRE_PROCESS_YUV420|Unmapped
||EXTRA_ENDING|Unmapped
Gather|GATHER|Mapped
||TILE|Unmapped
||GROUPED_CONV2D|Unmapped
||TOPK|Unmapped
||PRE_PROCESS_BGRA|Unmapped
LogicalNot|LOGICAL_NOT|Mapped
Sin|SIN|Mapped
Log|LOG|Mapped
ArgMin|ARGMIN|Mapped
||ROI_ALIGN|Unmapped
||HEATMAP_MAX_KEYPOINT|Unmapped
||AXIS_ALIGNED_BBOX_TRANSFORM|Unmapped
||BOX_WITH_NMS_LIMIT|Unmapped
||GENERATE_PROPOSALS|Unmapped
||DETECTION_POSTPROCESS|Unmapped
||RANDOM_MULTINOMIAL|Unmapped
||LOG_SOFTMAX|Unmapped
||RELU_KERAS|Unmapped
||GRU_OVXLIB|Unmapped
||GRUCELL_OVXLIB|Unmapped
||UNIDIRECTIONAL_SEQUENCE_RNN|Unmapped
||QUANTIZED_16BIT_LSTM|Unmapped
||BIDIRECTIONAL_SEQUENCE_RNN|Unmapped
||BIDIRECTIONAL_SEQUENCE_LSTM|Unmapped
||RNNCELL_OVXLIB|Unmapped
HardSwish|SWISH|Mapped
||DEPTHWISE_CONV1D|Unmapped
GatherNd|GATHER_ND|Mapped
Cast|CAST|Mapped
||LINEAR|Unmapped
||BATCHNORM_SINGLE|Unmapped
||MOMENTS|Unmapped
Squeeze|SQUEEZE|Mapped
HardSigmoid|HARD_SIGMOID|Unmapped
Mish|MISH|Unmapped
||EXPAND_BROADCAST|Unmapped
||PRE_PROCESS_YUV444|Unmapped
||PRE_PROCESS_NV12|Unmapped
||SCATTER_ND|Unmapped
||DECONVOLUTION1D|Unmapped
||INTERP|Unmapped
||RESIZE_1D|Unmapped