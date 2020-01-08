//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

//
// Based on neon, Intel(R) Nervana(tm) reference deep learning framework.
// Ported from Python to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package acc

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
    "strings"
)

//
//    KernelArgument
//

type KernelArgument interface{}

//
//    Kernel building pass 1: Convert args []backends.Value -> typeArgs []TypeArg
//

type TypeArg interface{
    String() string
    Key() string
    Hash() string
}

type TensorTypeArg struct {
    indx int
    dtype base.Dtype
    takeAxis int
    unitShape bool
}

func(a *TensorTypeArg) Indx() int {
    return a.indx
}

func(a *TensorTypeArg) Dtype() base.Dtype {
    return a.dtype
}

func(a *TensorTypeArg) TakeAxis() int {
    return a.takeAxis
}

func(a *TensorTypeArg) UnitShape() bool {
    return a.unitShape
}

func(a *TensorTypeArg) String() string {
    return fmt.Sprintf("tensor(indx: %d, dtype: %s)", a.indx, a.dtype)
}

func(a *TensorTypeArg) Key() string {
    return fmt.Sprintf("T%d", a.indx)
}

func(a *TensorTypeArg) Hash() string {
    return fmt.Sprintf("T%d_%s_%d_%t", a.indx, a.dtype, a.takeAxis, a.unitShape)
}

type FloatTypeArg struct {
    indx int
}

func(a *FloatTypeArg) Indx() int {
    return a.indx
}

func(a *FloatTypeArg) String() string {
    return fmt.Sprintf("float(indx: %d)", a.indx)
}

func(a *FloatTypeArg) Key() string {
    return fmt.Sprintf("F%d", a.indx)
}

func(a *FloatTypeArg) Hash() string {
    return fmt.Sprintf("F%d", a.indx)
}

type SymbolTypeArg struct {
    indx int
}

func(a *SymbolTypeArg) Indx() int {
    return a.indx
}

func(a *SymbolTypeArg) String() string {
    return fmt.Sprintf("symbol(indx: %d)", a.indx)
}

func(a *SymbolTypeArg) Key() string {
    return fmt.Sprintf("S%d", a.indx)
}

func(a *SymbolTypeArg) Hash() string {
    return fmt.Sprintf("S%d", a.indx)
}

type OpTypeArg struct {
    opName backends.Op
    opCnt int
    rounding bool
    hotAxis int
    threads int
}

func(a *OpTypeArg) OpName() backends.Op {
    return a.opName
}

func(a *OpTypeArg) OpCnt() int {
    return a.opCnt
}

func(a *OpTypeArg) Rounding() bool {
    return a.rounding
}

func(a *OpTypeArg) HotAxis() int {
    return a.hotAxis
}

func(a *OpTypeArg) Threads() int {
    return a.threads
}

func(a *OpTypeArg) String() string {
    return fmt.Sprintf("op(name: %s, indx: %d)", a.opName, a.opCnt)
}

func(a *OpTypeArg) Key() string {
    return fmt.Sprintf("O%d", a.opCnt)
}

func(a *OpTypeArg) Hash() string {
    return fmt.Sprintf("O%s_%d_%t_%d_%d", a.opName, a.opCnt, a.rounding, a.hotAxis, a.threads)
}

func HashTypeArgs(typeArgs []TypeArg) string {
    h := make([]string, len(typeArgs))
    for i, arg := range typeArgs {
        h[i] = arg.Hash()
    }
    return strings.Join(h, "_")
}

func CompileCompoundKernel(randState DeviceAllocation, args []backends.Value) (
        []TypeArg, []KernelArgument, *AccTensor, int, int, bool) {
    var out *AccTensor
    argCnt := 0
    opCnt := 0
    arrayIds := make(map[*AccTensor]int)
    constIds := make(map[float64]int)
    symbolIds := make(map[string]int)
    var kernelArgs []KernelArgument
    var typeArgs []TypeArg
    var shapeStack [][]int
    threads := 32
    redDepth := 0
    // Apply reduction constraints and determine thread axis
    // Blocks will be allocated counter to this axis
    // Also detect if this is a broadcast or transpose op.
    contiguous := true
    reduction := false
    broadcast := false
    transpose := false
    argminmax := false
    takeop := false
    axis := 1
    outShape := args[0].Shape()

    kernelArgsAppend := func(x KernelArgument) {
        kernelArgs = append(kernelArgs, x)
    }
    typeArgsAppend := func(x TypeArg) {
        typeArgs = append(typeArgs, x)
    } 
    shapeStackAppend := func(x []int) {
        shapeStack = append(shapeStack, x)
    }
    shapeStackPop := func() []int {
        n := len(shapeStack)
        shape := shapeStack[n-1]
        shapeStack = shapeStack[:n-1]
        return shape
    }

    if randState != nil {
        kernelArgsAppend(randState)
    }

    for _, arg := range args {
        switch v := arg.(type) {
        case *backends.OpTreeNode:
            opName := v.Op()
            switch {
            case isReductionOp(opName):
                if opName == backends.Argmax || opName == backends.Argmin {
                    argminmax = true
                }
                // To reduce a whole tensor (axis=None) reduce along each axis
                // in succession.
                argAxis := v.Axis()
                if argAxis != 0 && argAxis != 1 {
                    base.ValueError("Only reduction along an axis currently supported")
                }
                // Keep axis values consistent within the same kernel
                if reduction {
                    if argAxis != axis {
                        base.ValueError("Reduction only allowed along one axis per kernel.")
                    }
                } else {
                    reduction = true
                    axis = argAxis
                }
            case opName == backends.Onehot:
                takeop = true
            }

        case *AccTensor:
            argShape := v.Shape()
            switch {
            case len(argShape) < 2:
                broadcast = true
            case len(argShape) == 2 && 
                    base.IntsMin(argShape) == 1 && !base.IntsEq(argShape, outShape):
                broadcast = true
            case v.IsTrans():
                transpose = true
            case v.TakeArray() != nil:
                takeop = true
            case !v.IsContiguous():
                contiguous = false
            }
        }
    }

    // If reducing along axis 0 we need to reverse all strides.
    // Each block gets a column and the threads work down the columns.
    orderStrides := func(x []int) []int {
        if axis == 0 {
            return base.IntsReverse(x)
        } else {
            return base.IntsCopy(x)
        }
    }

    // ACHTUNG: Moved here to make maxShape visible for async call in the end
    maxShape := []int{1, 1}

    for _, arg := range args {
        switch v := arg.(type) {
        case *AccTensor:
            // Array operand
            argShape := v.Shape()
            var shape []int
            var strides []int
            if broadcast || reduction || transpose || takeop || !contiguous {
                // for complex operations, use the native dimensions
                if len(argShape) == 2 {
                    shape = v.Shape()
                    strides = orderStrides(v.Strides())
                } else {
                    base.ValueError(
                        "Operations that are not simple elementwise are only "+
                        "currently supported in 2 dimensions.")
                }
            } else {
                // use more efficient 2d dimensions if this is a plain ew op.
                shape, strides = getFastEwDims(v.Size())
                strides = orderStrides(strides)
            }

            // If same array is passed in multiple times to expression,
            // consolidate them into one kernel argument.
            indx, ok := arrayIds[v]
            if !ok {
                // The first array passed in should be the output.
                // It's ok if this array is duplicated as the first instance
                // needs to be a mutable pointer.
                // A subsequent instance of out (if present) will be a const pointer.
                if out == nil {
                    out = v
                    indx = argCnt
                } else {
                    arrayIds[v] = argCnt
                    indx = argCnt
                }
                argCnt++

                // support broadcast
                // Need to use shape of base array to determine stride if this
                // operation is a take
                if v.TakeArray() != nil {
                    baseShape := v.Base().Shape()
                    if baseShape[0] == 1 {
                        strides[1-axis] = 0
                    }
                    if baseShape[1] == 1 {
                        strides[axis] = 0
                    }
                } else {
                    if shape[0] == 1 {
                        strides[1-axis] = 0
                    }
                    if shape[1] == 1 {
                        strides[axis] = 0
                    }
                }

                kernelArgsAppend(v.AccData()) 
                kernelArgsAppend(strides[0])
                kernelArgsAppend(strides[1])

                // fancy indexing/take
                if takeArray := v.TakeArray(); takeArray != nil {
                    kernelArgsAppend(takeArray.Indices().AccData())
                }
            }

            // swap the take axis when reducing axis=0
            // also add 1 to distinguish between no take operations
            takeAxis := 0
            if takeArray := v.TakeArray(); takeArray != nil {
                if axis != 1 {
                    takeAxis = 2 - takeArray.Axis()
                } else {
                    takeAxis = takeArray.Axis() + 1
                }
            }
            // else no take operation

            typeArgsAppend(&TensorTypeArg{indx, v.Dtype(), takeAxis, (shape[axis]==1)})
            shapeStackAppend(shape)

        case *backends.Int:
            value := float64(v.Value())
            indx, ok := constIds[value]
            if !ok {
                constIds[value] = argCnt
                indx = argCnt
                argCnt++
                kernelArgsAppend(value)
            }

            typeArgsAppend(&FloatTypeArg{indx})
            shapeStackAppend([]int{1, 1})
        
        case *backends.Float:
            value := v.Value()
            indx, ok := constIds[value]
            if !ok {
                constIds[value] = argCnt
                indx = argCnt
                argCnt++
                kernelArgsAppend(value)
            }

            typeArgsAppend(&FloatTypeArg{indx})
            shapeStackAppend([]int{1, 1})

        case *generators.IntSymbol:
            symbol := v.Symbol()
            indx, ok := symbolIds[symbol]
            if !ok {
                symbolIds[symbol] = argCnt
                indx = argCnt
                argCnt++
                kernelArgsAppend(symbol)
            }

            typeArgsAppend(&SymbolTypeArg{indx})
            shapeStackAppend([]int{1, 1})

        case *generators.FloatSymbol:
            symbol := v.Symbol()
            indx, ok := symbolIds[symbol]
            if !ok {
                symbolIds[symbol] = argCnt
                indx = argCnt
                argCnt++
                kernelArgsAppend(symbol)
            }

            typeArgsAppend(&SymbolTypeArg{indx})
            shapeStackAppend([]int{1, 1})

        case *backends.OpTreeNode:
            opName := v.Op()

            if numOps, ok := isFloatOp(opName); ok {
                // we need to do the shape arithemtic for the current operation
                maxShape = []int{1, 1}
                for opNum := 0; opNum < numOps; opNum++ {
                    shape := shapeStackPop()
                    for i := 0; i < 2; i++ {
                        if shape[i] != maxShape[i] {
                            // support broadcast
                            // TODO(orig): don't allow output tensor itself to be broadcastable.
                            // The final output is fine as a broadcast, for example
                            // assigning a constant.
                            // You just dont want a tensor being assigned to a
                            // smaller shape.
                            if shape[i] == 1 || maxShape[i] == 1 {
                                maxShape[i] = base.IntMax(maxShape[i], shape[i])
                            } else {
                                base.TypeError("Input shape: %v not compatible", shape)
                            }
                        }
                    }
                }

                switch opName {
                case backends.Assign:
                    // the axis dim is the thread loop stop condition
                    kernelArgsAppend(maxShape[axis])

                    rounding := out.Rounding()

                    // support rounding to arbitrary mantissa size
                    if rounding != 0 {
                        switch {
                        case rounding < 0:
                            // convert bool to some default mantissa
                            rounding = 10
                        case out.Dtype() == base.Float32:
                            rounding = base.IntMin(rounding, 15)
                        case out.Dtype() == base.Float16:
                            rounding = base.IntMin(rounding, 10)
                        }
                        kernelArgsAppend(base.IntMax(rounding, 1))
                    }

                    // TODO: Following code is not platform-agnoctis: move to higher order package?
                    // speed up deep reduction by using more than 32 threads
                    if !argminmax {
                        switch {
                        case reduction:
                            if redDepth >= 256 {
                                threads = 64
                            }

                            // TODO(orig)
                            // Try to bring this code back after figuring out race conditions
                            // switch {
                            // case redDepth >= 4096:
                            //     threads = 1024
                            // case redDepth >= 2048:
                            //     threads = 512
                            // case redDepth >= 1024:
                            //     threads = 256
                            // case redDepth >= 512:
                            //     threads = 128
                            // case redDepth >= 256:
                            //     threads = 64
                            // }

                        case !(reduction || transpose) && maxShape[1] >= 512:
                            threads = 256
                        }
                    }

                    typeArgsAppend(&OpTypeArg{
                        opName: opName, 
                        opCnt: opCnt, 
                        rounding: (rounding > 0), 
                        threads: threads,
                    })

                case backends.Onehot:
                    // flip the onehot axis if reducing axis = 0
                    hotAxis := v.Axis()
                    if axis == 0 {
                        hotAxis = 1 - hotAxis
                    }
                    typeArgsAppend(&OpTypeArg{opName: opName, opCnt: opCnt, hotAxis: hotAxis})
                    shapeStackAppend(maxShape)
                    kernelArgsAppend(v.Idx().(*AccTensor).AccData())

                default:
                    typeArgsAppend(&OpTypeArg{opName: opName, opCnt: opCnt})
                    shapeStackAppend(maxShape)
                }
            
            } else if isReductionOp(opName) {
                shape := base.IntsCopy(shapeStackPop())

                redDepth = base.IntMax(redDepth, shape[axis])

                // Allow a new axis size if doing post reduction broadcast.
                // So we need to know the axis size prior to reduction.
                kernelArgsAppend(shape[axis])
                typeArgsAppend(&OpTypeArg{opName: opName, opCnt: opCnt})

                // reduce the current shape
                shape[axis] = 1

                // udpate the current shape state
                shapeStackAppend(shape)

            } else {
                base.TypeError("%d is not a valid operation", opName)
            }

            opCnt++

        default:
            base.TypeError("args must be instance of AccTensor, int, float, or OpTreeNode")
        }
    }

    return typeArgs, kernelArgs, out, threads, maxShape[1-axis], reduction
}

var fastEwDimsCache = make(map[int][2][]int)

func getFastEwDims(size int) ([]int, []int) {
    if dims, ok := fastEwDimsCache[size]; ok {
        return dims[0], dims[1]
    }

    shape, strides := getFastEwDimsRaw(size)

    fastEwDimsCache[size] = [2][]int{shape, strides}
    return shape, strides
}

func getFastEwDimsRaw(size int) ([]int, []int) {
    // TODO(orig): I can probably do much better than this code below,
    //     but I think most tensors are evenly divisable by 256 off the bat.
    ewSize := 256
    for ewSize > 0 {
        if size % ewSize == 0 {
            break
        }
        ewSize -= 32
    }
    if ewSize == 0 {
        ewSize = 255
        for ewSize > 0 {
            if size % ewSize == 0 {
                break
            }
            ewSize--
        }
    }
    shape := []int{size/ewSize, ewSize}
    return shape, ContiguousStrides(shape)
}

func isFloatOp(op backends.Op) (int, bool) {
    if !backends.IsFloatOp(op) {
        return 0, false
    }
    var numOps int
    switch {
    case backends.IsZeroOperandOp(op):
        numOps = 0
    case backends.IsUnaryOp(op):
        numOps = 1
    case backends.IsBinaryOp(op):
        numOps = 2
    default:
        base.AssertionError("invalid float op: %d", op)
    }
    return numOps, true
}

func isReductionOp(op backends.Op) bool {
    return backends.IsReductionOp(op)
}

//
//    Platform-agnostic components of kernel building pass 2
//

type TreeNode struct {
    arg TypeArg
    isTree bool
    isScalar bool
    redCount int
    leftChild *TreeNode
    rightChild *TreeNode
}

func(n *TreeNode) String() string {
    var left string
    if n.leftChild != nil {
        left = n.leftChild.String()
    } else {
        left = "nil"
    }
    var right string
    if n.rightChild != nil {
        right = n.rightChild.String()
    } else {
        right = "nil"
    }
    return fmt.Sprintf("node(%s, %s, %s)", n.arg, left, right)
}

type Stage struct {
    kind string
    stack []TypeArg
}

func(s *Stage) Kind() string {
    return s.kind
}

func(s *Stage) Stack() []TypeArg {
    return s.stack
}

func BuildStages(typeArgs []TypeArg) []Stage {
    // from the stack, rebuild a mutable tree
    tree := BuildTree(typeArgs)

    // split all reductions and post reduction scalar operations out of the tree
    // sub-trees are converted to stacks and pushed onto stages list
    stages := SplitStages(tree, nil, nil, nil, nil)

    // convert the remainder of tree to stack
    stages = append(stages, BuildLastStage(tree))

    return stages
}

func BuildTree(typeArgs []TypeArg) *TreeNode {
    var stack []*TreeNode
    stackPop := func() *TreeNode {
        n := len(stack)
        node := stack[n-1]
        stack = stack[:n-1]
        return node
    }
    for _, arg := range typeArgs {
        switch v := arg.(type) {
        case *OpTypeArg:
            argType := v.opName

            if numOps, ok := isFloatOp(argType); ok {
                // ops with zero args default to non-scalar
                node := &TreeNode{arg: arg, isTree: true, isScalar: (numOps > 0), redCount: 0}
                for i := 0; i < numOps; i++ {
                    operand := stackPop()
                    switch w := operand.arg.(type) {
                    case *OpTypeArg:
                        // if child is another node in the tree:
                        // accumulate reduction count
                        node.redCount += operand.redCount
                        // if a child is not scalar, then neither is this node
                        if !operand.isScalar {
                            node.isScalar = false
                        }
                    case *TensorTypeArg:
                        // if child is an input tensor (an output tensor has id=0) then
                        // this node is not scalar
                        // if it's an output tensor check the shape[axis]==1 flag
                        if w.indx > 0 || !w.unitShape {
                            node.isScalar = false
                        }
                    }
                    // children are added in reverse order
                    node.rightChild = node.leftChild
                    node.leftChild = operand
                }
                stack = append(stack, node)

            } else if isReductionOp(argType) {
                operand := stackPop()
                reds := 1
                // if child is another node accumulate reduction count
                if operand.isTree {
                    reds += operand.redCount
                }
                // reductions are scalar by definition
                stack = append(stack, &TreeNode{
                    arg: arg, 
                    isTree: true, 
                    isScalar: true, 
                    redCount: reds, 
                    leftChild: operand,
                })

            } else {
                base.TypeError("%d is not a valid operation", argType)
            }

        case *FloatTypeArg, *SymbolTypeArg, *TensorTypeArg:
            // tensors and scalars just get added to the stack
            // for later processing with operators
            stack = append(stack, &TreeNode{arg: arg, isTree: false})
        }
    }

    // the stack should now contain just a single node which is the complete tree
    return stack[0]
}

func SplitStages(
        node *TreeNode,
        duplicates map[string]TypeArg,
        aliases map[TypeArg]bool,
        stages []Stage,
        parents []*TreeNode) []Stage {
    // init data structures
    if duplicates == nil {
        duplicates = make(map[string]TypeArg)
        aliases = make(map[TypeArg]bool)
    }

    if node.isTree {
        arg := node.arg.(*OpTypeArg)
        opName := arg.opName

        // don't count assignment node as a parent,
        // it will always exist in the final stage which is processed outside of
        // this function
        if opName != backends.Assign {
            parents = append(parents, node)
        }

        // post order traversal (pulls the stages deepest in the tree first)
        if child := node.leftChild; child != nil {
            stages = SplitStages(child, duplicates, aliases, stages, parents)
        }
        if child := node.rightChild; child != nil {
            stages = SplitStages(child, duplicates, aliases, stages, parents)
        }

        if n := len(parents); n > 0 {
            parents = parents[:n-1] // pop
        }

        if isReductionOp(opName) {
            redStack := ProcessNode(node, aliases, duplicates)
            if redStack != nil {
                // add this reduction stack to the stages
                stages = append(stages, Stage{"reduction", redStack})
            }

            // decrement reduction count for all parents
            for _, parent := range parents {
                parent.redCount--
            }

            // walk up the parent list
            // TODO(orig): potentially do this iteratively to find longest common set
            // of operations
            var scalarParent *TreeNode
            n := len(parents)
            for i := n - 1; i >= 0; i-- {
                // find the highest parent that is both scalar and has no other
                // child reductions
                parent := parents[i]
                if parent.isScalar && parent.redCount == 0 {
                    scalarParent = parent
                } else {
                    break
                }
            }

            // if there are any scalar operations over this reduction, remove
            // them from the tree as well
            if scalarParent != nil {
                scalarStack := ProcessNode(scalarParent, aliases, duplicates)
                if scalarStack != nil {
                    // add this scalar stack to the stages
                    stages = append(stages, Stage{"scalar", scalarStack})
                }
            }
        }
    }

    return stages
}

func ProcessNode(
        node *TreeNode, 
        aliases map[TypeArg]bool, 
        duplicates map[string]TypeArg) []TypeArg {
    // generate a unique key from the stack of everything below this reduction
    stack := PostOrder(node, nil)
    var keys []string
    for _, item := range stack {
        // for operations, just append the name
        // aliases require the id as well since they encapsulate specific
        // tensors and constants
        // For alias append opCnt, for tensor or constant append id.
        var k string
        switch v := item.(type) {
        case *OpTypeArg:
            if _, ok := aliases[item]; !ok {
                k = fmt.Sprintf("O{%d}", v.opName)
            } else {
                k = fmt.Sprintf("O{%d %d}", v.opName, v.opCnt)
            }
        case *FloatTypeArg:
            k = fmt.Sprintf("F{%d}", v.indx)
        case *SymbolTypeArg:
            k = fmt.Sprintf("S{%d}", v.indx)
        case *TensorTypeArg:
            k = fmt.Sprintf("T{%d}", v.indx)
        }
        keys = append(keys, k)
    }
    key := fmt.Sprintf("%v", keys)

    // use the generated key to look for duplicates
    if dupNode, ok := duplicates[key]; ok {
        // if this is a duplicate, replace the stack with the op node of the
        // original reduction
        node.arg = dupNode
        // no new stage is returned in this case, the node is just converted
        // into an alias
        stack = nil
    } else {
        // first time seeing this reduction, record it in the dict
        // the last item in the stack will be the reduction op
        top := stack[len(stack)-1]
        duplicates[key] = top
        // record which nodes can be aliased
        aliases[top] = true
    }

    // drop any children
    node.leftChild = nil
    node.rightChild = nil

    return stack
}

func BuildLastStage(tree *TreeNode) Stage {
    // set the final stage type to type of output (scalar or elementwise)
    var lastStage string
    if tree.isScalar {
        lastStage = "red_out"
    } else {
        lastStage = "ew_out"
    }
    return Stage{lastStage, PostOrder(tree, nil)}
}

func PostOrder(node *TreeNode, stack []TypeArg) []TypeArg {
    if node.isTree {
        if child := node.leftChild; child != nil {
            stack = PostOrder(child, stack)
        }
        if child := node.rightChild; child != nil {
            stack = PostOrder(child, stack)
        }
    }
    stack = append(stack, node.arg)
    return stack
}

