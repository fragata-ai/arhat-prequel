//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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

package main

/*
A tool that shuffles training images in Arhat canned image data set.
Shuffling is essential for successful training. This tool shall be used
for post-processing of canned image data sets produced by Arhat ingestion
tools, in particular, Caltech256 and Imagenet.

By default, it converts files 'train_x.raw' and 'train_y.raw' into shuffled
versions 'train_x.dat' and 'train_y.dat'. (The user is expected to rename
respectively the files produced by the ingestion tools before shuffling;
we might provide a better automated data preparation pipeline in the future.)
*/

import (
    "encoding/binary"
    "fmt"
    "log"
    "math/rand"
    "os"
)

// configurable

var (
    inputX = "./data/train_x.raw"
    inputY = "./data/train_y.raw"
    outputX = "./data/train_x.dat"
    outputY = "./data/train_y.dat"
    imgSize = 3 * 224 * 224 // in bytes
)

func main() {
    err := Run()
    if err != nil {
        log.Fatal(err)
    }
}

func Run() error {
    var err error

    // open images
    fmt.Printf("Open images\n")
    xinfp, err := os.Open(inputX)
    if err != nil {
        return err
    }
    var xhdr [4]uint32
    err = binary.Read(xinfp, binary.LittleEndian, xhdr[:])
    if err != nil {
        return err
    }
    gotImgSize := int(xhdr[1]) * int(xhdr[2]) * int(xhdr[3])
    if gotImgSize != imgSize {
        err = fmt.Errorf("Image size mismatch: want %d got %d", imgSize, gotImgSize)
        return err
    }
    count := int(xhdr[0])
    fmt.Printf("  %d images\n", count)

    // read labels
    fmt.Printf("Read labels\n")
    defer xinfp.Close()
    yinfp, err := os.Open(inputY)
    if err != nil {
        return err
    }
    defer yinfp.Close()
    var yhdr [2]uint32
    err = binary.Read(yinfp, binary.LittleEndian, yhdr[:])
    if err != nil {
        return err
    }
    if xhdr[0] != yhdr[0] {
        err = fmt.Errorf("Count mismatch: images %d, labels %d", xhdr[0], yhdr[0])
        return err
    }
    yin := make([]uint32, count)
    err = binary.Read(yinfp, binary.LittleEndian, yin)
    if err != nil {
        return err
    }

    // define permutation
    perm := rand.Perm(count)

    // shuffle images
    fmt.Printf("Shuffle images\n")
    xoutfp, err := os.Create(outputX)
    if err != nil {
        return err
    }
    defer xoutfp.Close()
    err = binary.Write(xoutfp, binary.LittleEndian, xhdr[:])
    if err != nil {
        return err
    }
    xbuf := make([]uint8, imgSize)
    for i := 0; i < count; i++ {
        if (i + 1) % 1000 == 0 {
            fmt.Printf("  %d\n", i+1)
        }
        offs := int64(perm[i]) * int64(imgSize) + 16
        _, err = xinfp.Seek(offs, 0)
        if err != nil {
            return err
        }
        n, err := xinfp.Read(xbuf)
        if err != nil {
            return err
        }
        if n != imgSize {
            err = fmt.Errorf("Invalid read size: want %d got %d", imgSize, n)
        }
        n, err = xoutfp.Write(xbuf)
        if err != nil {
            return err
        }
        if n != imgSize {
            err = fmt.Errorf("Invalid write size: want %d got %d", imgSize, n)
        }
    }

    // shuffle labels
    fmt.Printf("Shuffle labels\n")
    yout := make([]uint32, count)
    for i := 0; i < count; i++ {
        yout[i] = yin[perm[i]]
    }

    // write labels
    fmt.Printf("Write labels\n")
    youtfp, err := os.Create(outputY)
    if err != nil {
        return err
    }
    defer youtfp.Close()
    err = binary.Write(youtfp, binary.LittleEndian, yhdr[:])
    if err != nil {
        return err
    }
    err = binary.Write(youtfp, binary.LittleEndian, yout)
    if err != nil {
        return err
    }

    return nil
}

