//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
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
Ingestion tool for the ILSVRC2012 data set. Converts the original TAR files into
Arhat canned image format. The image files are resized and cropped to 224x224.
The data are split into the training and validation sets based on a percentage 
ratio specified via the 'trainPct' variable.

After the conversion 'train_x.dat' and 'train_y.dat'
must be shuffled using the program available at 

fragata/arhat/tools/imagenet/shuffle_canned

The number of input synsets can be restructed by setting the 'numCat' variable.

The original data shall be represented as a collection of 1000 TAR files, 
one file per synset. The list of synsets shall be provided in 'list_2012.lst'.
The original data can be either obtained from the Imagenet site or gathered
using the program available at

fragata/arhat/tools/imagenet/gather_synset

The original data set is described at

http://www.image-net.org/challenges/LSVRC/2012/
*/

import (
    "archive/tar"
    "encoding/binary"
    "fmt"
    "image"
    "image/jpeg"
    "io"
    "io/ioutil"
    "log"
    "math/rand"
    "os"
    "path/filepath"
    "strings"

    "nfnt/resize"
)

// configurable

var (
    listPath = "./list_2012.lst"
    dataInput = "./raw"
    dataOutput = "./data"
    numCat = 100 // number of categories to be ingested; use 1000 for full set
    trainPct = 90
    testResize = false // if true, write resized images to testOutput
    testOutput = "./test"
)

func main() {
    err := Run()
    if err != nil {
        log.Fatal(err)
    }
}

func Run() error {
    var err error

    // read list
    data, err := ioutil.ReadFile(listPath)
    if err != nil {
        return err
    }
    list := strings.Split(string(data), "\n")

    // start output
    err = os.MkdirAll(dataOutput, 0777)
    if err != nil {
        return err
    }
    trainFn := filepath.Join(dataOutput, "train_x.dat")
    trainFp, err := os.Create(trainFn)
    if err != nil {
        return err
    }
    defer trainFp.Close()
    valFn := filepath.Join(dataOutput, "valid_x.dat")
    valFp, err := os.Create(valFn)
    if err != nil {
        return err
    }
    defer valFp.Close()
    var trainCat []uint32
    var valCat []uint32

    // reserve headers
    err = WriteHeaderX(trainFp, 0 /*numTrain*/, 224, 224, 3)
    if err != nil {
        return err
    }
    err = WriteHeaderX(valFp, 0 /*numVal*/, 224, 224, 3)
    if err != nil {
        return err
    }

    // process categories
    cat := 0
    for _, line := range list {
        if cat >= numCat {
            break
        }
        flds := strings.SplitN(line, " ", 2)
        trainCat, valCat, err = RunCat(cat, flds[0], trainFp, valFp, trainCat, valCat)
        if err != nil {
            return err
        }
        cat++
    }

    // finalize headers
    _, err = trainFp.Seek(0, 0)
    if err != nil {
        return err
    }
    err = WriteHeaderX(trainFp, len(trainCat), 224, 224, 3)
    if err != nil {
        return err
    }
    _, err = valFp.Seek(0, 0)
    if err != nil {
        return err
    }
    err = WriteHeaderX(valFp, len(valCat), 224, 224, 3)
    if err != nil {
        return err
    }

    // finalize output
    err = WriteCategories(trainCat, true)
    if err != nil {
        return err
    }
    err = WriteCategories(valCat, false)
    if err != nil {
        return err
    }

    return nil
}

func RunCat(
        cat int, 
        id string, 
        trainFp io.Writer, 
        valFp io.Writer, 
        trainCat []uint32, 
        valCat []uint32) ([]uint32, []uint32, error) {
    var err error

    sourceTar := dataInput + "/" + id + ".tar"

fmt.Printf("Processing %s\n", sourceTar)

    // split data
    split, err := SplitData(sourceTar)
    if err != nil {
        return nil, nil, err
    }

    numTrain := 0
    numVal := 0

    // scan source tar
    fp, err := os.Open(sourceTar)
    if err != nil {
        return nil, nil, err
    }
    defer fp.Close()
    tr := tar.NewReader(fp)
    for {
        hdr, err := tr.Next()
        if err == io.EOF {
            break
        }
        if err != nil {
            return nil, nil, err
        }
        name := hdr.Name
        switch split[name] {
        case 0:
            // bad image
        case 1:
            err = ProcessImage(tr, name, trainFp, true)
            if err != nil {
                return nil, nil, err
            }
            trainCat = append(trainCat, uint32(cat))
            numTrain++
        case 2:
            err = ProcessImage(tr, name, valFp, false)
            if err != nil {
                return nil, nil, err
            }
            valCat = append(valCat, uint32(cat))
            numVal++
        }
    }

fmt.Printf("Cat %d [%s]: train %d val %d\n", cat, id, numTrain, numVal)

    return trainCat, valCat, nil
}

func SplitData(sourceTar string) (map[string]int, error) {
    // build list
    var list []string
    fp, err := os.Open(sourceTar)
    if err != nil {
        return nil, err
    }
    defer fp.Close()
    tr := tar.NewReader(fp)
    for {
        hdr, err := tr.Next()
        if err == io.EOF {
            break
        }
        if err != nil {
            return nil, err
        }
        _, _, err = image.Decode(tr)
        if err != nil {
            fmt.Printf("Invalid image: %s\n", hdr.Name)
            continue
        }
        list = append(list, hdr.Name)
    }

    // split list to sets
    split := make(map[string]int)
    n := len(list)
    perm := rand.Perm(n)
    numTrain := (n * trainPct + 99) / 100
    for i := 0; i < n; i++ {
        k := perm[i]
        if i < numTrain {
            split[list[k]] = 1
        } else {
            split[list[k]] = 2
        }
    }
    return split, nil
}

func ProcessImage(r io.Reader, name string, w io.Writer, train bool) error {
    img, _, err := image.Decode(r)
    if err != nil {
        return err
    }

    bounds := img.Bounds()
    xmin := bounds.Min.X
    xmax := bounds.Max.X
    ymin := bounds.Min.Y
    ymax := bounds.Max.Y

/*
    fmt.Printf("  bbox %d %d %d %d\n", 
        bounds.Min.X, bounds.Max.X, bounds.Min.Y, bounds.Max.Y)
*/

    dx := xmax - xmin
    dy := ymax - ymin
    xoff := 0
    yoff := 0
    var size int
    if dx > dy {
        xoff = (dx - dy) / 2
        size = dy
    } else {
        yoff = (dy - dx) / 2
        size = dx
    }

    var tmpRect image.Rectangle
    tmpRect.Min.X = 0
    tmpRect.Max.X = size
    tmpRect.Min.Y = 0
    tmpRect.Max.Y = size
    tmp := image.NewRGBA(tmpRect)
    for x := 0; x < size; x++ {
        for y := 0; y < size; y++ {
            c := img.At(x+xoff, y+yoff)
            tmp.Set(x, y, c)
        }
    }

    out := resize.Resize(224, 224, tmp, resize.Bilinear)

    err = WriteImage(out, 224, w)
    if err != nil {
        return err
    }

    if testResize {
        err = TestWrite(out, name, train)
        if err != nil {
            return err
        }
    }

    return nil
}

func WriteImage(img image.Image, size int, w io.Writer) error {
    n := size * size
    buf := make([]uint8, n*3)
    k := 0
    // [CHW]
    for y := 0; y < size; y++ {
        for x := 0; x < size; x++ {
            c := img.At(x, y)
            r, g, b, _ := c.RGBA()
            buf[k] = uint8(r)
            buf[k+n] = uint8(g)
            buf[k+2*n] = uint8(b)
            k++
        }
    }
    _, err := w.Write(buf)
    if err != nil {
        return err
    }
    if err != nil {
        return err
    }
    return nil
}

func TestWrite(img image.Image, name string, train bool) error {
    var sub string
    if train {
        sub = filepath.Join(testOutput, "train")
    } else {
        sub = filepath.Join(testOutput, "val")
    }
    err := os.MkdirAll(sub, 0777)
    if err != nil {
        return err
    }
    path := filepath.Join(sub, name)
    fp, err := os.Create(path)
    if err != nil {
        return err
    }
    defer fp.Close()
    err = jpeg.Encode(fp, img, nil)
    if err != nil {
        return err
    }
    return nil
}

func WriteCategories(cat []uint32, train bool) error {
    var fn string
    if train {
        fn = filepath.Join(dataOutput, "train_y.dat")
    } else {
        fn = filepath.Join(dataOutput, "valid_y.dat")
    }
    fp, err := os.Create(fn)
    if err != nil {
        return err
    }
    defer fp.Close()
    err = WriteHeaderY(fp, len(cat), numCat)
    if err != nil {
        return err
    }
    err = binary.Write(fp, binary.LittleEndian, cat)
    if err != nil {
        return err
    }
    return nil
}

func WriteHeaderX(w io.Writer, numImg int, height int, width int, numChan int) error {
    var hdr [4]uint32
    hdr[0] = uint32(numImg)
    hdr[1] = uint32(height)
    hdr[2] = uint32(width)
    hdr[3] = uint32(numChan)
    err := binary.Write(w, binary.LittleEndian, hdr[:])
    if err != nil {
        return err
    }
    return nil
}

func WriteHeaderY(w io.Writer, numImg int, numCat int) error {
    var hdr [2]uint32
    hdr[0] = uint32(numImg)
    hdr[1] = uint32(numCat)
    err := binary.Write(w, binary.LittleEndian, hdr[:])
    if err != nil {
        return err
    }
    return nil
}

