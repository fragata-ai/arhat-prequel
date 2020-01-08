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

//
// Based on NumPy linalg/lapack_lite package distributed under the following license.
// For details, see https://github.com/numpy/numpy/tree/master/numpy/linalg/lapack_lite
//

/*
Copyright (c) 1992-2013 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved.
Copyright (c) 2000-2013 The University of California Berkeley. All
                        rights reserved.
Copyright (c) 2006-2013 The University of Colorado Denver.  All rights
                        reserved.

$COPYRIGHT$

Additional copyrights may follow

$HEADER$

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

The copyright holders provide no reassurances that the source code
provided does not infringe any patent, copyright, or any other
intellectual property rights of third parties.  The copyright holders
disclaim any liability to any recipient for claims brought against
recipient by any third party for infringement of that parties
intellectual property rights.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "runtime_cpu/arhat.h"

namespace arhat {
namespace cpu {

inline int max(int x, int y) {
    return (x >= y) ? x : y;
}

inline bool lsame(int x, int y) {
    if (x == y) {
        return true;
    }
    if (x >= 'a' && x <= 'z') {
        x = (x - 'a') + 'A';
    }
    if (y >= 'a' && y <= 'z') {
        y = (y - 'a') + 'A';
    }
    return (x == y);
}

/*
    Purpose
    =======

    SGEMM  performs one of the matrix-matrix operations

       C := alpha * op(A) * op(B) + beta * C,

    where  op(X) is one of

       op(X) = X or op(X) = X',

    alpha and beta are scalars, and A, B and C are matrices, with op(A)
    an m by k matrix, op(B) a k by n matrix and C an m by n matrix.

    Arguments
    ==========

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op(A) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n',  op(A) = A

                TRANSA = 'T' or 't',  op(A) = A'

                TRANSA = 'C' or 'c',  op(A) = A'

             Unchanged on exit.

    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op(B) to be used in
             the matrix multiplication as follows:

                TRANSB = 'N' or 'n',  op(B) = B

                TRANSB = 'T' or 't',  op(B) = B'

                TRANSB = 'C' or 'c',  op(B) = B'

             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix
             op( A )  and of the  matrix  C.  M  must  be at least  zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix
             op( B ) and the number of columns of the matrix C. N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix
             op( A ) and the number of rows of the matrix op( B ). K must
             be at least  zero.
             Unchanged on exit.

    ALPHA  - REAL            .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by m  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When  TRANSA = 'N' or 'n' then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, k ).
             Unchanged on exit.

    B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             LDB must be at least  max( 1, k ), otherwise  LDB must be at
             least  max( 1, n ).
             Unchanged on exit.

    BETA   - REAL            .
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             supplied as zero then C need not be set on input.
             Unchanged on exit.

    C      - REAL             array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, m ).
             Unchanged on exit.

    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

    =====================================================================

*/

int BlasSgemm(
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        float alpha, 
        const float *a, 
        int lda, 
        const float *b, 
        int ldb, 
        float beta, 
        float *c, 
        int ldc) {
    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i1, i2, i3;

    //
    //    Set NOTA and NOTB as true if A and B respectively are not
    //    transposed and set NROWA, NCOLA and NROWB as the number of rows
    //    and columns of A and the number of rows of B respectively
    //

    // Local variables
    int i, j, l, info;
    bool nota, notb;
    float temp;
    int ncola;
    int nrowa, nrowb;
 
    // Parameter adjustments
    a_dim1 = lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = ldc;
    c_offset = 1 + c_dim1;
    c -= c_offset;

    // Function Body
    nota = lsame(transa, 'N');
    notb = lsame(transb, 'N');
    if (nota) {
        nrowa = m;
        ncola = k;
    } else {
        nrowa = k;
        ncola = m;
    }
    if (notb) {
        nrowb = k;
    } else {
        nrowb = n;
    }

     //
     //    Test the input parameters
     //

    info = 0;
    if (!nota && !lsame(transa, 'C') && !lsame(transa, 'T')) {
        info = 1;
    } else if (!notb && !lsame(transb, 'C') && !lsame(transb, 'T')) {
        info = 2;
    } else if (m < 0) {
        info = 3;
    } else if (n < 0) {
        info = 4;
    } else if (k < 0) {
        info = 5;
    } else if (lda < max(1, nrowa)) {
        info = 8;
    } else if (ldb < max(1, nrowb)) {
        info = 10;
    } else if (ldc < max(1, m)) {
        info = 13;
    }
    if (info != 0) {
        return info;
    }

    //
    //    Quick return if possible
    //

    if (m == 0 || n == 0 || (alpha == 0.0f || k == 0) && beta == 1.0f) {
        return 0;
    }

    //
    //    And if alpha == 0
    //

    if (alpha == 0.0f) {
        if (beta == 0.0f) {
            i1 = n;
            for (j = 1; j <= i1; j++) {
                i2 = m;
                for (i = 1; i <= i2; i++) {
                    c[i + j * c_dim1] = 0.0f;
                }
            }
        } else {
            i1 = n;
            for (j = 1; j <= i1; j++) {
                i2 = m;
                for (i = 1; i <= i2; i++) {
                    c[i + j * c_dim1] = beta * c[i + j * c_dim1];
                }
            }
        }
        return 0;
    }

    //
    //    Start the operations.
    //

    if (notb) {
        if (nota) {
            //
            //    Form C := alpha * A * B + beta * C
            //
            i1 = n;
            for (j = 1; j <= i1; j++) {
                if (beta == 0.0f) {
                    i2 = m;
                    for (i = 1; i <= i2; i++) {
                        c[i + j * c_dim1] = 0.0f;
                    }
                } else if (beta != 1.0f) {
                    i2 = m;
                    for (i = 1; i <= i2; i++) {
                        c[i + j * c_dim1] = beta * c[i + j * c_dim1];
                    }
                }
                i2 = k;
                for (l = 1; l <= i2; l++) {
                    if (b[l + j * b_dim1] != 0.0f) {
                        temp = alpha * b[l + j * b_dim1];
                        i3 = m;
                        for (i = 1; i <= i3; i++) {
                            c[i + j * c_dim1] += temp * a[i + l * a_dim1];
                        }
                    }
                }
            }
        } else {
            //
            //    Form C := alpha * A' * B + beta * C
            //
            i1 = n;
            for (j = 1; j <= i1; j++) {
                i2 = m;
                for (i = 1; i <= i2; i++) {
                    temp = 0.0f;
                    i3 = k;
                    for (l = 1; l <= i3; l++) {
                        temp += a[l + i * a_dim1] * b[l + j * b_dim1];
                    }
                    if (beta == 0.0f) {
                        c[i + j * c_dim1] = alpha * temp;
                    } else {
                        c[i + j * c_dim1] = alpha * temp + beta * c[i + j * c_dim1];
                    }
                }
            }
        }
    } else {
        if (nota) {
            //
            //    Form C := alpha * A * B' + beta * C
            //
            i1 = n;
            for (j = 1; j <= i1; j++) {
                if (beta == 0.0f) {
                    i2 = m;
                    for (i = 1; i <= i2; i++) {
                        c[i + j * c_dim1] = 0.0f;
                    }
                } else if (beta != 1.0f) {
                    i2 = m;
                    for (i = 1; i <= i2; i++) {
                        c[i + j * c_dim1] = beta * c[i + j * c_dim1];
                    }
                }
                i2 = k;
                for (l = 1; l <= i2; l++) {
                    if (b[j + l * b_dim1] != 0.0f) {
                        temp = alpha * b[j + l * b_dim1];
                        i3 = m;
                        for (i = 1; i <= i3; i++) {
                            c[i + j * c_dim1] += temp * a[i + l * a_dim1];
                        }
                    }
                }
            }
        } else {
            //
            //    Form C := alpha * A' * B' + beta * C
            //
            i1 = n;
            for (j = 1; j <= i1; j++) {
                i2 = m;
                for (i = 1; i <= i2; i++) {
                    temp = 0.0f;
                    i3 = k;
                    for (l = 1; l <= i3; l++) {
                        temp += a[l + i * a_dim1] * b[j + l * b_dim1];
                    }
                    if (beta == 0.0f) {
                        c[i + j * c_dim1] = alpha * temp;
                    } else {
                        c[i + j * c_dim1] = alpha * temp + beta * c[i + j * c_dim1];
                    }
                }
            }
        }
    }

    return 0;
}

/*
    Purpose
    =======

    SGEMV  performs one of the matrix-vector operations

       y := alpha * A * x + beta * y, or y := alpha * A' * x + beta * y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ==========

    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   y := alpha * A * x + beta * y

                TRANS = 'T' or 't'   y := alpha * A' * x + beta * y

                TRANS = 'C' or 'c'   y := alpha * A' * x + beta * y

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - REAL            .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - REAL             array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

    X      - REAL             array of DIMENSION at least
             ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
             Before entry, the incremented array X must contain the
             vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA   - REAL            .
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Y      - REAL             array of DIMENSION at least
             ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
             Before entry with BETA non-zero, the incremented array Y
             must contain the vector y. On exit, Y is overwritten by the
             updated vector y.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================
*/

int BlasSgemv(
        int trans, 
        int m, 
        int n, 
        float alpha,
        const float *a, 
        int lda, 
        const float *x, 
        int incx, 
        float beta, 
        float *y,
        int incy) {
    // System generated locals
    int a_dim1, a_offset, i1, i2;

    // Local variables
    int i, j, ix, iy, jx, jy, kx, ky, info;
    float temp;
    int lenx, leny;

    // Parameter adjustments
    a_dim1 = lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    x--;
    y--;

    // Function Body
    info = 0;
    if (!lsame(trans, 'N') && !lsame(trans, 'T') && !lsame(trans, 'C')) {
        info = 1;
    } else if (m < 0) {
        info = 2;
    } else if (n < 0) {
        info = 3;
    } else if (lda < max(1, m)) {
        info = 6;
    } else if (incx == 0) {
        info = 8;
    } else if (incy == 0) {
        info = 11;
    }
    if (info != 0) {
        return info;
    }

    //
    //    Quick return if possible
    //
    if (m == 0 || n == 0 || alpha == 0.0f && beta == 1.0f) {
        return 0;
    }

    //
    //    Set  LENX  and  LENY, the lengths of the vectors x and y, and set
    //    up the start points in X and Y
    //

    if (lsame(trans, 'N')) {
        lenx = n;
        leny = m;
    } else {
        lenx = m;
        leny = n;
    }
    if (incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (lenx - 1) * incx;
    }
    if (incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (leny - 1) * incy;
    }

    //
    //    Start the operations. In this version the elements of A are
    //    accessed sequentially with one pass through A
    //
    //    First form y := beta * y
    //

    if (beta != 1.0f) {
        if (incy == 1) {
            if (beta == 0.0f) {
                i1 = leny;
                for (i = 1; i <= i1; i++) {
                    y[i] = 0.0f;
                }
            } else {
                i1 = leny;
                for (i = 1; i <= i1; i++) {
                    y[i] = beta * y[i];
                }
            }
        } else {
            iy = ky;
            if (beta == 0.0f) {
                i1 = leny;
                for (i = 1; i <= i1; i++) {
                    y[iy] = 0.0f;
                    iy += incy;
                }
            } else {
                i1 = leny;
                for (i = 1; i <= i1; i++) {
                    y[iy] = beta * y[iy];
                    iy += incy;
                }
            }
        }
    }
    if (alpha == 0.0f) {
        return 0;
    }
    if (lsame(trans, 'N')) {
        //
        //    Form y := alpha * A * x + y
        //
        jx = kx;
        if (incy == 1) {
            i1 = n;
            for (j = 1; j <= i1; j++) {
                if (x[jx] != 0.0f) {
                    temp = alpha * x[jx];
                    i2 = m;
                    for (i = 1; i <= i2; i++) {
                        y[i] += temp * a[i + j * a_dim1];
                    }
                }
                jx += incx;
            }
        } else {
            i1 = n;
            for (j = 1; j <= i1; j++) {
                if (x[jx] != 0.0f) {
                    temp = alpha * x[jx];
                    iy = ky;
                    i2 = m;
                    for (i = 1; i <= i2; i++) {
                        y[iy] += temp * a[i + j * a_dim1];
                        iy += incy;
                    }
                }
                jx += incx;
            }
        }
    } else {
        //
        //    Form y := alpha * A' * x + y
        //
        jy = ky;
        if (incx == 1) {
            i1 = n;
            for (j = 1; j <= i1; j++) {
                temp = 0.0f;
                i2 = m;
                for (i = 1; i <= i2; i++) {
                    temp += a[i + j * a_dim1] * x[i];
                }
                y[jy] += alpha * temp;
                jy += incy;
            }
        } else {
            i1 = n;
            for (j = 1; j <= i1; j++) {
                temp = 0.0f;
                ix = kx;
                i2 = m;
                for (i = 1; i <= i2; i++) {
                    temp += a[i + j * a_dim1] * x[ix];
                    ix += incx;
                }
                y[jy] += alpha * temp;
                jy += incy;
            }
        }
    }

    return 0;
}

} // cpu
} // arhat

