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
A tool that gathers Imagenet images for a given synset ID.
Original image URLs provided by Imagenet are used. 
Images that are not availble anymore at provided URLs are skipped.

Image URLs are obtained using the method described at

http://image-net.org/download-imageurls
*/

import (
    "archive/tar"
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
    "strings"
    "time"
)

var outDir = "./raw" // configurable

func main() {
    if len(os.Args) != 2 {
        fmt.Fprintf(os.Stderr, "Usage: imgnet_gather_synset <id>\n")
        os.Exit(1)
    }
    err := GatherSynset(os.Args[1])
    if err != nil {
        fmt.Fprintf(os.Stderr, "%s\n", err.Error())
    }
}

func GatherSynset(synset string) error {
    tmo, err := time.ParseDuration("1m")
    if err != nil {
        return err
    }
    http.DefaultClient.Timeout = tmo
    urls, err := GetImageUrls(synset)
    if err != nil {
        return err
    }
    lfn := synset + ".log"
    lfp, err := os.Create(lfn)
    if err != nil {
        return err
    }
    defer lfp.Close()
    tfn := outDir + "/" + synset + ".tar"
    tfp, err := os.Create(tfn)
    if err != nil {
        return err
    }
    defer tfp.Close()
    tw := tar.NewWriter(tfp)
    defer tw.Close()    
    var meta []string
    total := 0
    index := 0
    for _, url := range urls {
        if len(url) == 0 {
            continue
        }
        url := strings.TrimSpace(url)
        var sfx string
        switch {
        case strings.HasSuffix(url, ".jpg") || strings.HasSuffix(url, ".JPG") || 
                strings.HasSuffix(url, ".jpeg") || strings.HasSuffix(url, ".JPEG"):
            sfx = "jpg"
        case strings.HasSuffix(url, ".png") || strings.HasSuffix(url, ".PNG"):
            sfx = "png"
        default:
            fmt.Fprintf(lfp, "Unsupported image type: %s\n", url)
            continue
        }
        total++
        fmt.Fprintf(lfp, "Loading %s\n", url)
        data, err := GetImage(url)
        if err != nil {
            fmt.Fprintf(lfp, "%s\n", err.Error())
            continue
        }
        sdata := string(data)
        if strings.Contains(sdata, "html>") || strings.Contains(sdata, "HTML>") {
            fmt.Fprintf(lfp, "HTML response detected\n")
            continue
        }
        index++
        meta = append(meta, fmt.Sprintf("%04d.%s %s", index, sfx, url))
/* for testing only
        err = WriteImage(synset, index, sfx, data)
        if err != nil {
            return err
        }
*/
        err = WriteTarImage(tw, synset, index, sfx, data)
        if err != nil {
            return err
        }
    }
    fmt.Fprintf(lfp, "Synset %s: have %d of %d\n", synset, index, total)
    mfn := synset + ".meta"
    mfp, err := os.Create(mfn)
    if err != nil {
        return err
    }
    defer mfp.Close()
    for _, ml := range meta {
        fmt.Fprintf(mfp, "%s\n", ml)
    }
    return nil
}

// e.g. n02107683 = Berner mountain dog
func GetImageUrls(synset string) ([]string, error) {
    url := "http://image-net.org/api/text/imagenet.synset.geturls?wnid=" + synset
    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    data, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }
    return strings.Split(string(data), "\n"), nil
}

func GetImage(url string) ([]byte, error) {
    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    if resp.StatusCode != 200 {
        err = fmt.Errorf("Response status code %d", resp.StatusCode)
        return nil, err
    }
    data, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }
    return data, nil
}

func WriteTarImage(tw *tar.Writer, synset string, index int, sfx string, data []byte) error {
    fn := fmt.Sprintf("%s/%04d.%s", synset, index, sfx)
    hdr := &tar.Header{
        Name: fn,
        Mode: 0666,
        Size: int64(len(data)),
    }
    err := tw.WriteHeader(hdr)
    if err != nil {
        return err
    }
    _, err = tw.Write(data)
    if err != nil {
        return err
    }
    return nil
}

func WriteImage(synset string, index int, sfx string, data []byte) error {
    dir := outDir + "/" + synset
    err := os.MkdirAll(dir, 0777)
    if err != nil {
        return err
    }
    fn := dir + fmt.Sprintf("/%04d.%s", index, sfx)
    err = ioutil.WriteFile(fn, data, 0666)
    if err != nil {
        return err
    }
    return nil
}

