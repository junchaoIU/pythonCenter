??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8??
x
dense_3/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes

:
p
dense_3/biasVarHandleOp*
shape:*
shared_namedense_3/bias*
dtype0*
_output_shapes
: 
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:
|
dense_1_1/kernelVarHandleOp*
shape
:
*!
shared_namedense_1_1/kernel*
dtype0*
_output_shapes
: 
u
$dense_1_1/kernel/Read/ReadVariableOpReadVariableOpdense_1_1/kernel*
dtype0*
_output_shapes

:

t
dense_1_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*
shared_namedense_1_1/bias
m
"dense_1_1/bias/Read/ReadVariableOpReadVariableOpdense_1_1/bias*
dtype0*
_output_shapes
:

|
dense_2_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:
*!
shared_namedense_2_1/kernel
u
$dense_2_1/kernel/Read/ReadVariableOpReadVariableOpdense_2_1/kernel*
dtype0*
_output_shapes

:

t
dense_2_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namedense_2_1/bias
m
"dense_2_1/bias/Read/ReadVariableOpReadVariableOpdense_2_1/bias*
dtype0*
_output_shapes
:
|
training_1/Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *%
shared_nametraining_1/Adam/iter
u
(training_1/Adam/iter/Read/ReadVariableOpReadVariableOptraining_1/Adam/iter*
dtype0	*
_output_shapes
: 
?
training_1/Adam/beta_1VarHandleOp*'
shared_nametraining_1/Adam/beta_1*
shape: *
_output_shapes
: *
dtype0
y
*training_1/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_1/Adam/beta_1*
dtype0*
_output_shapes
: 
?
training_1/Adam/beta_2VarHandleOp*
shape: *'
shared_nametraining_1/Adam/beta_2*
dtype0*
_output_shapes
: 
y
*training_1/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_1/Adam/beta_2*
dtype0*
_output_shapes
: 
~
training_1/Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *&
shared_nametraining_1/Adam/decay*
shape: 
w
)training_1/Adam/decay/Read/ReadVariableOpReadVariableOptraining_1/Adam/decay*
dtype0*
_output_shapes
: 
?
training_1/Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining_1/Adam/learning_rate*
shape: 
?
1training_1/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_1/Adam/learning_rate*
dtype0*
_output_shapes
: 
u
true_positivesVarHandleOp*
_output_shapes
: *
shared_nametrue_positives*
dtype0*
shape:?
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
dtype0*
_output_shapes	
:?
u
true_negativesVarHandleOp*
_output_shapes
: *
shared_nametrue_negatives*
dtype0*
shape:?
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
dtype0*
_output_shapes	
:?
w
false_positivesVarHandleOp* 
shared_namefalse_positives*
shape:?*
dtype0*
_output_shapes
: 
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
dtype0*
_output_shapes	
:?
w
false_negativesVarHandleOp*
_output_shapes
: * 
shared_namefalse_negatives*
dtype0*
shape:?
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
?
 training_1/Adam/dense_3/kernel/mVarHandleOp*
shape
:*1
shared_name" training_1/Adam/dense_3/kernel/m*
dtype0*
_output_shapes
: 
?
4training_1/Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp training_1/Adam/dense_3/kernel/m*
dtype0*
_output_shapes

:
?
training_1/Adam/dense_3/bias/mVarHandleOp*/
shared_name training_1/Adam/dense_3/bias/m*
shape:*
dtype0*
_output_shapes
: 
?
2training_1/Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOptraining_1/Adam/dense_3/bias/m*
dtype0*
_output_shapes
:
?
"training_1/Adam/dense_1_1/kernel/mVarHandleOp*
_output_shapes
: *3
shared_name$"training_1/Adam/dense_1_1/kernel/m*
dtype0*
shape
:

?
6training_1/Adam/dense_1_1/kernel/m/Read/ReadVariableOpReadVariableOp"training_1/Adam/dense_1_1/kernel/m*
_output_shapes

:
*
dtype0
?
 training_1/Adam/dense_1_1/bias/mVarHandleOp*1
shared_name" training_1/Adam/dense_1_1/bias/m*
shape:
*
dtype0*
_output_shapes
: 
?
4training_1/Adam/dense_1_1/bias/m/Read/ReadVariableOpReadVariableOp training_1/Adam/dense_1_1/bias/m*
_output_shapes
:
*
dtype0
?
"training_1/Adam/dense_2_1/kernel/mVarHandleOp*
_output_shapes
: *3
shared_name$"training_1/Adam/dense_2_1/kernel/m*
dtype0*
shape
:

?
6training_1/Adam/dense_2_1/kernel/m/Read/ReadVariableOpReadVariableOp"training_1/Adam/dense_2_1/kernel/m*
dtype0*
_output_shapes

:

?
 training_1/Adam/dense_2_1/bias/mVarHandleOp*
_output_shapes
: *
shape:*1
shared_name" training_1/Adam/dense_2_1/bias/m*
dtype0
?
4training_1/Adam/dense_2_1/bias/m/Read/ReadVariableOpReadVariableOp training_1/Adam/dense_2_1/bias/m*
_output_shapes
:*
dtype0
?
 training_1/Adam/dense_3/kernel/vVarHandleOp*1
shared_name" training_1/Adam/dense_3/kernel/v*
shape
:*
_output_shapes
: *
dtype0
?
4training_1/Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp training_1/Adam/dense_3/kernel/v*
_output_shapes

:*
dtype0
?
training_1/Adam/dense_3/bias/vVarHandleOp*/
shared_name training_1/Adam/dense_3/bias/v*
shape:*
dtype0*
_output_shapes
: 
?
2training_1/Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOptraining_1/Adam/dense_3/bias/v*
dtype0*
_output_shapes
:
?
"training_1/Adam/dense_1_1/kernel/vVarHandleOp*
_output_shapes
: *
shape
:
*3
shared_name$"training_1/Adam/dense_1_1/kernel/v*
dtype0
?
6training_1/Adam/dense_1_1/kernel/v/Read/ReadVariableOpReadVariableOp"training_1/Adam/dense_1_1/kernel/v*
_output_shapes

:
*
dtype0
?
 training_1/Adam/dense_1_1/bias/vVarHandleOp*
_output_shapes
: *1
shared_name" training_1/Adam/dense_1_1/bias/v*
dtype0*
shape:

?
4training_1/Adam/dense_1_1/bias/v/Read/ReadVariableOpReadVariableOp training_1/Adam/dense_1_1/bias/v*
_output_shapes
:
*
dtype0
?
"training_1/Adam/dense_2_1/kernel/vVarHandleOp*
shape
:
*3
shared_name$"training_1/Adam/dense_2_1/kernel/v*
dtype0*
_output_shapes
: 
?
6training_1/Adam/dense_2_1/kernel/v/Read/ReadVariableOpReadVariableOp"training_1/Adam/dense_2_1/kernel/v*
dtype0*
_output_shapes

:

?
 training_1/Adam/dense_2_1/bias/vVarHandleOp*
_output_shapes
: *
shape:*1
shared_name" training_1/Adam/dense_2_1/bias/v*
dtype0
?
4training_1/Adam/dense_2_1/bias/v/Read/ReadVariableOpReadVariableOp training_1/Adam/dense_2_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?*
ConstConst"/device:CPU:0*?*
value?)B?) B?)
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratemHmImJmKmLmMvNvOvPvQvRvS
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
?
&layer_regularization_losses

'layers
(metrics
)non_trainable_variables
	variables
regularization_losses
trainable_variables
 
 
 
 
?
*layer_regularization_losses

+layers
,metrics
-non_trainable_variables
	variables
regularization_losses
trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
.layer_regularization_losses

/layers
0metrics
1non_trainable_variables
	variables
regularization_losses
trainable_variables
\Z
VARIABLE_VALUEdense_1_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_1_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
2layer_regularization_losses

3layers
4metrics
5non_trainable_variables
	variables
regularization_losses
trainable_variables
\Z
VARIABLE_VALUEdense_2_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_2_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
6layer_regularization_losses

7layers
8metrics
9non_trainable_variables
	variables
regularization_losses
trainable_variables
SQ
VARIABLE_VALUEtraining_1/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_1/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_1/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_1/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_1/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

:0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
;
thresholds
<true_positives
=true_negatives
>false_positives
?false_negatives
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/0/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/0/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/0/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/0/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
>2
?3
 
 
?
Dlayer_regularization_losses

Elayers
Fmetrics
Gnon_trainable_variables
@	variables
Aregularization_losses
Btrainable_variables
 
 
 

<0
=1
>2
?3
??
VARIABLE_VALUE training_1/Adam/dense_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_1/Adam/dense_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_1/Adam/dense_1_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_1/Adam/dense_1_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_1/Adam/dense_2_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_1/Adam/dense_2_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_1/Adam/dense_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_1/Adam/dense_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_1/Adam/dense_1_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_1/Adam/dense_1_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_1/Adam/dense_2_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_1/Adam/dense_2_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
~
serving_default_dense_inputPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense_3/kerneldense_3/biasdense_1_1/kerneldense_1_1/biasdense_2_1/kerneldense_2_1/bias*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*
Tout
2*+
f&R$
"__inference_signature_wrapper_5070*
Tin
	2*+
_gradient_op_typePartitionedCall-5256
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp$dense_1_1/kernel/Read/ReadVariableOp"dense_1_1/bias/Read/ReadVariableOp$dense_2_1/kernel/Read/ReadVariableOp"dense_2_1/bias/Read/ReadVariableOp(training_1/Adam/iter/Read/ReadVariableOp*training_1/Adam/beta_1/Read/ReadVariableOp*training_1/Adam/beta_2/Read/ReadVariableOp)training_1/Adam/decay/Read/ReadVariableOp1training_1/Adam/learning_rate/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp4training_1/Adam/dense_3/kernel/m/Read/ReadVariableOp2training_1/Adam/dense_3/bias/m/Read/ReadVariableOp6training_1/Adam/dense_1_1/kernel/m/Read/ReadVariableOp4training_1/Adam/dense_1_1/bias/m/Read/ReadVariableOp6training_1/Adam/dense_2_1/kernel/m/Read/ReadVariableOp4training_1/Adam/dense_2_1/bias/m/Read/ReadVariableOp4training_1/Adam/dense_3/kernel/v/Read/ReadVariableOp2training_1/Adam/dense_3/bias/v/Read/ReadVariableOp6training_1/Adam/dense_1_1/kernel/v/Read/ReadVariableOp4training_1/Adam/dense_1_1/bias/v/Read/ReadVariableOp6training_1/Adam/dense_2_1/kernel/v/Read/ReadVariableOp4training_1/Adam/dense_2_1/bias/v/Read/ReadVariableOpConst*&
f!R
__inference__traced_save_5304*(
Tin!
2	*
_output_shapes
: *+
_gradient_op_typePartitionedCall-5305**
config_proto

GPU 

CPU2J 8*
Tout
2
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasdense_1_1/kerneldense_1_1/biasdense_2_1/kerneldense_2_1/biastraining_1/Adam/itertraining_1/Adam/beta_1training_1/Adam/beta_2training_1/Adam/decaytraining_1/Adam/learning_ratetrue_positivestrue_negativesfalse_positivesfalse_negatives training_1/Adam/dense_3/kernel/mtraining_1/Adam/dense_3/bias/m"training_1/Adam/dense_1_1/kernel/m training_1/Adam/dense_1_1/bias/m"training_1/Adam/dense_2_1/kernel/m training_1/Adam/dense_2_1/bias/m training_1/Adam/dense_3/kernel/vtraining_1/Adam/dense_3/bias/v"training_1/Adam/dense_1_1/kernel/v training_1/Adam/dense_1_1/bias/v"training_1/Adam/dense_2_1/kernel/v training_1/Adam/dense_2_1/bias/v*'
Tin 
2*)
f$R"
 __inference__traced_restore_5398*
_output_shapes
: *+
_gradient_op_typePartitionedCall-5399**
config_proto

GPU 

CPU2J 8*
Tout
2??
?	
?
)__inference_sequential_layer_call_fn_5144

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5043*
Tin
	2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-5044**
config_proto

GPU 

CPU2J 8*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5097

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0\

dense/ReluReludense/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0`
dense_1/ReluReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
?
?
"__inference_signature_wrapper_5070
dense_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tin
	2*+
_gradient_op_typePartitionedCall-5061*(
f#R!
__inference__wrapped_model_4894*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :+ '
%
_user_specified_namedense_input: : 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5000
dense_input(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4917*
Tin
2*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4911*
Tout
2?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*
Tin
2*
Tout
2*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4939*+
_gradient_op_typePartitionedCall-4945?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*
Tout
2*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_4967*+
_gradient_op_typePartitionedCall-4973?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall: : : : :+ '
%
_user_specified_namedense_input: : 
?!
?
__inference__wrapped_model_4894
dense_input3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
v
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
sequential/dense_2/SigmoidSigmoid#sequential/dense_2/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitysequential/dense_2/Sigmoid:y:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp: : : : :+ '
%
_user_specified_namedense_input: : 
?	
?
)__inference_sequential_layer_call_fn_5133

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6**
config_proto

GPU 

CPU2J 8*+
_gradient_op_typePartitionedCall-5017*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5016*
Tout
2*
Tin
	2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
?
$__inference_dense_layer_call_fn_5162

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4911**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-4917?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
&__inference_dense_1_layer_call_fn_5180

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4945**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4939*
Tout
2*
Tin
2*'
_output_shapes
:?????????
?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_5191

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????
::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?l
?
 __inference__traced_restore_5398
file_prefix#
assignvariableop_dense_3_kernel#
assignvariableop_1_dense_3_bias'
#assignvariableop_2_dense_1_1_kernel%
!assignvariableop_3_dense_1_1_bias'
#assignvariableop_4_dense_2_1_kernel%
!assignvariableop_5_dense_2_1_bias+
'assignvariableop_6_training_1_adam_iter-
)assignvariableop_7_training_1_adam_beta_1-
)assignvariableop_8_training_1_adam_beta_2,
(assignvariableop_9_training_1_adam_decay5
1assignvariableop_10_training_1_adam_learning_rate&
"assignvariableop_11_true_positives&
"assignvariableop_12_true_negatives'
#assignvariableop_13_false_positives'
#assignvariableop_14_false_negatives8
4assignvariableop_15_training_1_adam_dense_3_kernel_m6
2assignvariableop_16_training_1_adam_dense_3_bias_m:
6assignvariableop_17_training_1_adam_dense_1_1_kernel_m8
4assignvariableop_18_training_1_adam_dense_1_1_bias_m:
6assignvariableop_19_training_1_adam_dense_2_1_kernel_m8
4assignvariableop_20_training_1_adam_dense_2_1_bias_m8
4assignvariableop_21_training_1_adam_dense_3_kernel_v6
2assignvariableop_22_training_1_adam_dense_3_bias_v:
6assignvariableop_23_training_1_adam_dense_1_1_kernel_v8
4assignvariableop_24_training_1_adam_dense_1_1_bias_v:
6assignvariableop_25_training_1_adam_dense_2_1_kernel_v8
4assignvariableop_26_training_1_adam_dense_2_1_bias_v
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/0/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/0/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/0/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/0/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
RestoreV2/shape_and_slicesConst"/device:CPU:0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*)
dtypes
2	*?
_output_shapesn
l:::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:{
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_1_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_2_1_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_2_1_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp'assignvariableop_6_training_1_adam_iterIdentity_6:output:0*
dtype0	*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0?
AssignVariableOp_7AssignVariableOp)assignvariableop_7_training_1_adam_beta_1Identity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0?
AssignVariableOp_8AssignVariableOp)assignvariableop_8_training_1_adam_beta_2Identity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_training_1_adam_decayIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0?
AssignVariableOp_10AssignVariableOp1assignvariableop_10_training_1_adam_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_true_positivesIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_true_negativesIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_false_positivesIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_negativesIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp4assignvariableop_15_training_1_adam_dense_3_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype0P
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T0?
AssignVariableOp_16AssignVariableOp2assignvariableop_16_training_1_adam_dense_3_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_training_1_adam_dense_1_1_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_training_1_adam_dense_1_1_bias_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_training_1_adam_dense_2_1_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_training_1_adam_dense_2_1_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_training_1_adam_dense_3_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOp2assignvariableop_22_training_1_adam_dense_3_bias_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_training_1_adam_dense_1_1_kernel_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_training_1_adam_dense_1_1_bias_vIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_training_1_adam_dense_2_1_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype0P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_training_1_adam_dense_2_1_bias_vIdentity_26:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0?
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_28Identity_28:output:0*?
_input_shapesp
n: :::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262$
AssignVariableOpAssignVariableOp: : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : : : :
 
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_5173

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_4985
dense_input(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4917*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4911*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:??????????
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4945*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4939*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4973*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_4967*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:??????????
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall: : : : :+ '
%
_user_specified_namedense_input: : 
?<
?
__inference__traced_save_5304
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop/
+savev2_dense_1_1_kernel_read_readvariableop-
)savev2_dense_1_1_bias_read_readvariableop/
+savev2_dense_2_1_kernel_read_readvariableop-
)savev2_dense_2_1_bias_read_readvariableop3
/savev2_training_1_adam_iter_read_readvariableop	5
1savev2_training_1_adam_beta_1_read_readvariableop5
1savev2_training_1_adam_beta_2_read_readvariableop4
0savev2_training_1_adam_decay_read_readvariableop<
8savev2_training_1_adam_learning_rate_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop?
;savev2_training_1_adam_dense_3_kernel_m_read_readvariableop=
9savev2_training_1_adam_dense_3_bias_m_read_readvariableopA
=savev2_training_1_adam_dense_1_1_kernel_m_read_readvariableop?
;savev2_training_1_adam_dense_1_1_bias_m_read_readvariableopA
=savev2_training_1_adam_dense_2_1_kernel_m_read_readvariableop?
;savev2_training_1_adam_dense_2_1_bias_m_read_readvariableop?
;savev2_training_1_adam_dense_3_kernel_v_read_readvariableop=
9savev2_training_1_adam_dense_3_bias_v_read_readvariableopA
=savev2_training_1_adam_dense_1_1_kernel_v_read_readvariableop?
;savev2_training_1_adam_dense_1_1_bias_v_read_readvariableopA
=savev2_training_1_adam_dense_2_1_kernel_v_read_readvariableop?
;savev2_training_1_adam_dense_2_1_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_f8ea3cbbe8274d3cbd1a38f2b83ddc15/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/0/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/0/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/0/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/0/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop+savev2_dense_1_1_kernel_read_readvariableop)savev2_dense_1_1_bias_read_readvariableop+savev2_dense_2_1_kernel_read_readvariableop)savev2_dense_2_1_bias_read_readvariableop/savev2_training_1_adam_iter_read_readvariableop1savev2_training_1_adam_beta_1_read_readvariableop1savev2_training_1_adam_beta_2_read_readvariableop0savev2_training_1_adam_decay_read_readvariableop8savev2_training_1_adam_learning_rate_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop;savev2_training_1_adam_dense_3_kernel_m_read_readvariableop9savev2_training_1_adam_dense_3_bias_m_read_readvariableop=savev2_training_1_adam_dense_1_1_kernel_m_read_readvariableop;savev2_training_1_adam_dense_1_1_bias_m_read_readvariableop=savev2_training_1_adam_dense_2_1_kernel_m_read_readvariableop;savev2_training_1_adam_dense_2_1_bias_m_read_readvariableop;savev2_training_1_adam_dense_3_kernel_v_read_readvariableop9savev2_training_1_adam_dense_3_bias_v_read_readvariableop=savev2_training_1_adam_dense_1_1_kernel_v_read_readvariableop;savev2_training_1_adam_dense_1_1_bias_v_read_readvariableop=savev2_training_1_adam_dense_2_1_kernel_v_read_readvariableop;savev2_training_1_adam_dense_2_1_bias_v_read_readvariableop"/device:CPU:0*)
dtypes
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::
:
:
:: : : : : :?:?:?:?:::
:
:
::::
:
:
:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : : : : :
 
?
?
&__inference_dense_2_layer_call_fn_5198

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_4967*'
_output_shapes
:?????????*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_gradient_op_typePartitionedCall-4973?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_4911

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
)__inference_sequential_layer_call_fn_5026
dense_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tout
2*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5016**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-5017*
Tin
	2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :+ '
%
_user_specified_namedense_input: : 
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_4939

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5043

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*+
_gradient_op_typePartitionedCall-4917*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4911*
Tout
2*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*'
_output_shapes
:?????????
**
config_proto

GPU 

CPU2J 8*+
_gradient_op_typePartitionedCall-4945*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4939?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_4967*'
_output_shapes
:?????????*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_gradient_op_typePartitionedCall-4973?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5122

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0\

dense/ReluReludense/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0`
dense_1/ReluReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_4967

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_5155

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5016

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*
Tout
2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4911*
Tin
2*+
_gradient_op_typePartitionedCall-4917?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4939*
Tin
2**
config_proto

GPU 

CPU2J 8*+
_gradient_op_typePartitionedCall-4945*'
_output_shapes
:?????????
*
Tout
2?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tout
2*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_4967*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_gradient_op_typePartitionedCall-4973?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
)__inference_sequential_layer_call_fn_5053
dense_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tout
2*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5043**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tin
	2*+
_gradient_op_typePartitionedCall-5044?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :+ '
%
_user_specified_namedense_input: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
C
dense_input4
serving_default_dense_input:0?????????;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
T__call__
*U&call_and_return_all_conditional_losses
V_default_save_signature"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": [null, 15], "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": [null, 15], "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["AUC"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
W__call__
*X&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "dense_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 15], "config": {"batch_input_shape": [null, 15], "dtype": "float32", "sparse": false, "name": "dense_input"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 15], "config": {"name": "dense", "trainable": true, "batch_input_shape": [null, 15], "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
]__call__
*^&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}}
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratemHmImJmKmLmMvNvOvPvQvRvS"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
&layer_regularization_losses

'layers
(metrics
)non_trainable_variables
	variables
regularization_losses
trainable_variables
T__call__
V_default_save_signature
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
,
_serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
*layer_regularization_losses

+layers
,metrics
-non_trainable_variables
	variables
regularization_losses
trainable_variables
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
.layer_regularization_losses

/layers
0metrics
1non_trainable_variables
	variables
regularization_losses
trainable_variables
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1_1/kernel
:
2dense_1_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
2layer_regularization_losses

3layers
4metrics
5non_trainable_variables
	variables
regularization_losses
trainable_variables
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_2_1/kernel
:2dense_2_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
6layer_regularization_losses

7layers
8metrics
9non_trainable_variables
	variables
regularization_losses
trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_1/Adam/iter
 : (2training_1/Adam/beta_1
 : (2training_1/Adam/beta_2
: (2training_1/Adam/decay
':% (2training_1/Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?#
;
thresholds
<true_positives
=true_negatives
>false_positives
?false_negatives
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
`__call__
*a&call_and_return_all_conditional_losses"?!
_tf_keras_layer?!{"class_name": "AUC", "name": "AUC", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "AUC", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593]}}
 "
trackable_list_wrapper
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
<
<0
=1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dlayer_regularization_losses

Elayers
Fmetrics
Gnon_trainable_variables
@	variables
Aregularization_losses
Btrainable_variables
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
0:.2 training_1/Adam/dense_3/kernel/m
*:(2training_1/Adam/dense_3/bias/m
2:0
2"training_1/Adam/dense_1_1/kernel/m
,:*
2 training_1/Adam/dense_1_1/bias/m
2:0
2"training_1/Adam/dense_2_1/kernel/m
,:*2 training_1/Adam/dense_2_1/bias/m
0:.2 training_1/Adam/dense_3/kernel/v
*:(2training_1/Adam/dense_3/bias/v
2:0
2"training_1/Adam/dense_1_1/kernel/v
,:*
2 training_1/Adam/dense_1_1/bias/v
2:0
2"training_1/Adam/dense_2_1/kernel/v
,:*2 training_1/Adam/dense_2_1/bias/v
?2?
)__inference_sequential_layer_call_fn_5133
)__inference_sequential_layer_call_fn_5144
)__inference_sequential_layer_call_fn_5053
)__inference_sequential_layer_call_fn_5026?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_5122
D__inference_sequential_layer_call_and_return_conditional_losses_5000
D__inference_sequential_layer_call_and_return_conditional_losses_5097
D__inference_sequential_layer_call_and_return_conditional_losses_4985?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_4894?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
dense_input?????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_5162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_5155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_5180?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_5173?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_5198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_5191?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
5B3
"__inference_signature_wrapper_5070dense_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
D__inference_sequential_layer_call_and_return_conditional_losses_4985m<?9
2?/
%?"
dense_input?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_dense_2_layer_call_and_return_conditional_losses_5191\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ?
)__inference_sequential_layer_call_fn_5053`<?9
2?/
%?"
dense_input?????????
p 

 
? "???????????
"__inference_signature_wrapper_5070?C?@
? 
9?6
4
dense_input%?"
dense_input?????????"1?.
,
dense_2!?
dense_2?????????y
&__inference_dense_2_layer_call_fn_5198O/?,
%?"
 ?
inputs?????????

? "???????????
D__inference_sequential_layer_call_and_return_conditional_losses_5097h7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_5122h7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
__inference__wrapped_model_4894q4?1
*?'
%?"
dense_input?????????
? "1?.
,
dense_2!?
dense_2??????????
)__inference_sequential_layer_call_fn_5144[7?4
-?*
 ?
inputs?????????
p 

 
? "??????????w
$__inference_dense_layer_call_fn_5162O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_sequential_layer_call_and_return_conditional_losses_5000m<?9
2?/
%?"
dense_input?????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_dense_layer_call_and_return_conditional_losses_5155\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
)__inference_sequential_layer_call_fn_5133[7?4
-?*
 ?
inputs?????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_5026`<?9
2?/
%?"
dense_input?????????
p

 
? "??????????y
&__inference_dense_1_layer_call_fn_5180O/?,
%?"
 ?
inputs?????????
? "??????????
?
A__inference_dense_1_layer_call_and_return_conditional_losses_5173\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? 