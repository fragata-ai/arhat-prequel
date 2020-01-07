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
Ingestion tool for Caltech256 data set. Converts the original TAR file into
Arhat canned image format. The image files are resized and cropped to 224x224.
The data are split into the training and validation sets based on a percentage 
ratio specified via the 'trainPct' variable.

After the conversion 'train_x.dat' and 'train_y.dat'
must be shuffled using the program available at 

fragata/arhat/tools/imagenet/shuffle_canned

The original Caltech256 data set can be obtained at

http://www.vision.caltech.edu/Image_Datasets/Caltech256/
*/

import (
    "archive/tar"
    "encoding/binary"
    "fmt"
    "image"
    "image/jpeg"
    "io"
    "log"
    "math/rand"
    "os"
    "path/filepath"
    "strconv"
    "strings"

    "nfnt/resize"
)

// configurable

var (
    sourceTar = "256_ObjectCategories.tar"
    dataOutput = "./data"
    trainPct = 80
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

    // split data
    split, err := SplitData()
    if err != nil {
        return err
    }

    // count totals
    numTrain := 0
    numVal := 0
    for _, train := range split {
        if train {
            numTrain++
        } else {
            numVal++
        }
    }

    // write headers
    err = WriteHeaderX(trainFp, numTrain, 224, 224, 3)
    if err != nil {
        return err
    }
    err = WriteHeaderX(valFp, numVal, 224, 224, 3)
    if err != nil {
        return err
    }

    // scan source tar
    fp, err := os.Open(sourceTar)
    if err != nil {
        return err
    }
    defer fp.Close()
    tr := tar.NewReader(fp)
    for {
        hdr, err := tr.Next()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
        if !strings.HasSuffix(hdr.Name, ".jpg") {
            continue
        }
        name, cat, err := ParseImagePath(hdr.Name)
        if err != nil {
            return err
        }
        if cat > 255 {
            continue // skip extra categories
        }
        if split[name] {
            err = ProcessImage(tr, name, trainFp, true)
            if err != nil {
                return err
            }
            trainCat = append(trainCat, uint32(cat))
        } else {
            err = ProcessImage(tr, name, valFp, false)
            if err != nil {
                return err
            }
            valCat = append(valCat, uint32(cat))
        }
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

func SplitData() (map[string]bool, error) {
    // build catalog
    catMap := make(map[int][]string)
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
        if !strings.HasSuffix(hdr.Name, ".jpg") {
            continue
        }
        name, cat, err := ParseImagePath(hdr.Name)
        if err != nil {
            return nil, err
        }
        if cat > 255 {
            continue // skip extra categories
        }
        catMap[cat] = append(catMap[cat], name)
    }

    // split catalog to sets
    split := make(map[string]bool)
    for _, set := range catMap {
        n := len(set)
        perm := rand.Perm(n)
        numTrain := (n * trainPct + 99) / 100
        for i := 0; i < n; i++ {
            k := perm[i]
            split[set[k]] = (i < numTrain)
        }
    }
    return split, nil
}

func ParseImagePath(path string) (name string, cat int, err error) {
    parts := strings.Split(path, "/")
    if len(parts) != 3 {
        err = fmt.Errorf("Invalid image path: %s", path)
        return
    }
    name = parts[2]
    parts = strings.SplitN(parts[1], ".", 2)
    cat, err = strconv.Atoi(parts[0])
    if err != nil {
        return
    }
    cat--
    return
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

    fmt.Printf("  bbox %d %d %d %d\n", 
        bounds.Min.X, bounds.Max.X, bounds.Min.Y, bounds.Max.Y)

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
    // y axis points down: is this OK?
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
    err = WriteHeaderY(fp, len(cat), 256)
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

