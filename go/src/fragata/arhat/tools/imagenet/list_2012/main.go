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
A tool that gathers 1000 ILSVRC2012 synset IDs and descriptions
directly from the Imagenet Web site at

http://image-net.org/challenges/LSVRC/2012/browse-synsets

Outputs a list of synset IDs and descriptions as a text file.
Optionally generates gathering scripts for Windows and Linux.

The list of synset IDs without descriptions is also available at

http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list 
*/

import (
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
    "regexp"
)

// configurable

var (
    sourceUrl = "http://image-net.org/challenges/LSVRC/2012/browse-synsets"
    windowsScripts = false // generate gather scripts for Windows
    linuxScripts = false   // generate gather scripys for Linus
)

func main() {
    err := List()
    if err != nil {
        fmt.Fprintf(os.Stderr, "%s\n", err.Error())
        os.Exit(1)
    }
}

var reSynset = regexp.MustCompile(`<a href="http://image-net.org/synset\?wnid=(n\d*)">([^<]*)</a>`)

func List() error {
    resp, err := http.Get(sourceUrl)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    data, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return err
    }

    subs := reSynset.FindAllStringSubmatch(string(data), -1)
    lfp, err := os.Create("./list_2012.lst")
    if err != nil {
        return err
    }
    defer lfp.Close()
    count := len(subs)
    fmt.Printf("Synsets found: %d\n", count)
    for _, s := range subs {
        fmt.Fprintf(lfp, "%s %s\n", s[1], s[2])
    }

    if windowsScripts {
        index := 1
        for start := 0; start < count; start += 20 {
            stop := start + 20
            if stop > count {
                stop = count
            }
            bfn := fmt.Sprintf("./gather%02d.bat", index)
            bfp, err := os.Create(bfn)
            if err != nil {
                return err
            }
            for i := start; i < stop; i++ {
                // MS_DOS line format
                fmt.Fprintf(bfp, "bin\\imagenet_gather_synset.exe %s\r\n", subs[i][1])
            }
            bfp.Close()
            index++
        }
    }

    if linuxScripts {
        index := 1
        for start := 0; start < count; start += 20 {
            stop := start + 20
            if stop > count {
                stop = count
            }
            bfn := fmt.Sprintf("./gather%02d.sh", index)
            bfp, err := os.Create(bfn)
            if err != nil {
                return err
            }
            for i := start; i < stop; i++ {
                fmt.Fprintf(bfp, "echo \"%s\"\n", subs[i][1])
                fmt.Fprintf(bfp, "bin/imagenet_gather_synset %s\n", subs[i][1])
            }
            bfp.Close()
            index++
        }
    }

    return nil
}

