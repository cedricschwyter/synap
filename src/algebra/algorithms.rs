//! Functions of this module are meant to be 'raw' implementations of the respective algorithm. The
//! functions generally expect that all relevant checks (i.e., matrix/vector dimension validation)
//! are performed by the caller. This is in the hopes to be able to save computing resources during
//! actual execution of an algorithm.
//!
//! # NOTE:
//! If you are a client of the library it is recommended to use the methods on the [`Matrix`](super::matrix::Matrix) struct
//! that implement the required behaviour as these functions perform the necessary assertions and
//! generally choose the most appropriate algorithm for the job automatically.
//! Unless you wish to explicitely execute a specific algorithm, you are most likely looking in the
//! wrong place right now.

use num::{abs, pow, Float, Signed};
use std::sync::mpsc;
use std::{sync::Arc, thread};

use crate::algebra::matrix::*;

/// A straightforward, generic implementation of the euclidean scalar product
/// $\langle x, y \rangle = x^T y$ of two vectors $x, y$ that returns a scalar matrix (a 1 x 1 matrix) corresponding to the
/// numerical value of the scalar product. This implementation is numerically stable.
///
/// * `lhs` - corresponds to $x^T$ above, of size $1 \times d$
/// * `rhs` - corresponds to $y$ above, expected to be of size $d \times 1$
pub fn euclidean_scalar_product_naive<T: Field<T>, E: MatrixAccessor<T>>(
    lhs: &Matrix<T, E>,
    rhs: &Matrix<T, E>,
) -> Matrix<T, DenseAccessor<T>> {
    let mut value = lhs[0][0] * rhs[0][0];
    for i in 1..lhs.width() {
        value = value + lhs[0][i] * rhs[i][0];
    }
    Matrix::<T, E>::scalar(value)
}

/// A naive implementation of the euclidean norm
/// $\lvert \lvert x \rvert \rvert = \sqrt{\langle x, x \rangle}$ of a vector $x$ that returns a
/// scalar matrix (a 1 x 1 matrix) corresponding to the numerical value of the norm. This
/// implementation does not provide any guarantees about its numerical stability.
///
/// * `matrix` - corresponds to $x$ above, expected to be of size $\dim x \times 1$
pub fn euclidean_norm_naive<T: Field<T> + Float, E: MatrixAccessor<T>>(
    matrix: &Matrix<T, E>,
) -> Matrix<T, DenseAccessor<T>> {
    Matrix::<T, E>::scalar(
        euclidean_scalar_product_naive(matrix, matrix)
            .to_scalar()
            .sqrt(),
    )
}

/// A naive implementation of the more general $p$-norm: $\lvert \lvert x \rvert \rvert_p =
/// (\lvert x_1 \rvert ^ p + ... + \lvert x_n \rvert ^ p)^\frac{1}{p}$. This implementation does
/// not provide any guarantees about its numerical stability.
///
/// * `matrix` - corresponds to the $x$ above, expected to be of size $\dim x \times 1$
pub fn p_norm_naive<T: Field<T> + Float + Signed, E: MatrixAccessor<T>>(
    matrix: &Matrix<T, E>,
    p: usize,
) -> Matrix<f64, DenseAccessor<f64>> {
    let mut value: T = num::zero();
    for i in 0..matrix.height() {
        value = value + pow(abs(matrix[i][0]), p);
    }
    Matrix::<f64, DenseAccessor<f64>>::scalar(nth_root(value.to_f64().unwrap(), p as f64))
}

/// [https://rosettacode.org/wiki/Nth_root#Rust](https://rosettacode.org/wiki/Nth_root#Rust)
///
/// Solves for $x$ in $x^n = v$. This implementation does not provide any guarantees about its
/// numerical stability
///
/// * `value` - corresponds to $v$ above
/// * `n` - corresponds to $n$ above
pub fn nth_root(value: f64, n: f64) -> f64 {
    let p = 1e-9_f64;
    let mut x0 = value / n;
    loop {
        let x1 = ((n - 1.0) * x0 + value / f64::powf(x0, n - 1.0)) / n;
        if (x1 - x0).abs() < (x0 * p).abs() {
            return x1;
        };
        x0 = x1
    }
}

/// A naive implementation of the matrix multiplication algorithm, a classic $O(m^3)$
/// implementation.
/// It multiplies together matrices $A, B$ of sizes $n \times m$ and $m \times p$ respectively and
/// returns a matrix of size $n \times p$. This implementation is numerically stable.
///
/// * `lhs` - corresponds to $A$ above, expected to be of size $n \times m$
/// * `rhs` - corresponds to $B$ above, expected to be of size $m \times p$
pub fn mat_mul_naive<T: Field<T>, E: MatrixAccessor<T>>(
    lhs: &Matrix<T, E>,
    rhs: &Matrix<T, E>,
) -> Matrix<T, DenseAccessor<T>> {
    let mut elements = DenseAccessor::init(rhs.width(), lhs.height());
    for row in 0..lhs.height() {
        for col in 0..rhs.width() {
            let mut r = Vec::new();
            for i in 0..rhs.height() {
                r.push(rhs[i][col]);
            }
            elements[row][col] = (Matrix::<T, E>::vector(lhs[row].to_vec()).transpose()
                * Matrix::<T, E>::vector(r))
            .to_scalar();
        }
    }
    Matrix::new(elements)
}

/// A naive implementation of the matrix multiplication algorithm, a classic $O(m^3)$
/// implementation, running on as many threads as allowed by the host.
/// It multiplies together matrices $A, B$ of sizes $n \times m$ and $m \times p$ respectively and
/// returns a matrix of size $n \times p$. This implementation is numerically stable.
///
/// * `lhs` - corresponds to $A$ above, expected to be of size $n \times m$
/// * `rhs` - corresponds to $B$ above, expected to be of size $m \times p$
pub fn mat_mul_naive_threaded<T: Field<T> + Send + Sync + 'static, E: MatrixAccessor<T>>(
    lhs: &Matrix<T, E>,
    rhs: &Matrix<T, E>,
) -> Matrix<T, DenseAccessor<T>> {
    let mut res = DenseAccessor::init(rhs.width(), lhs.height());
    todo!();
    Matrix::new(res)
}

/// A classic, naive implementation of the gaussian row reduction algorithm, with runtime
/// complexity $O(n^3)$. Returns a new matrix that corresponds to the row-reduced variant of the
/// input matrix. Transforms the input matrix into row-echelon form. This implementation does not
/// provide any guarantees about its numerical stability.
///
/// * `matrix` - the matrix to reduce
pub fn gauss_elim_naive<T: Field<T>, E: MatrixAccessor<T>>(matrix: &Matrix<T, E>) -> Matrix<T, E> {
    let mut elements = matrix.elements();
    for i in 1..matrix.height() {
        for j in 0..i {
            if elements[i][j] != num::zero() {
                let r: T = elements[i][j] / elements[j][j];
                for k in 0..matrix.width() {
                    elements[i][k] = elements[i][k] / r - elements[j][k];
                }
            }
        }
    }
    Matrix::new(elements)
}

pub fn lu_decomp_naive<T: Field<T>, E: MatrixAccessor<T>>(
    matrix: &Matrix<T, E>,
) -> (Matrix<T, E>, Matrix<T, E>) {
    unimplemented!();
}

pub fn qr_decomp_naive<T: Field<T>, E: MatrixAccessor<T>>(
    matrix: &Matrix<T, E>,
) -> (Matrix<T, E>, Matrix<T, E>) {
    unimplemented!();
}

pub fn det_naive<T: Field<T>, E: MatrixAccessor<T>>(matrix: &Matrix<T, E>) -> T {
    unimplemented!();
}

pub fn rank_naive<T: Field<T>, E: MatrixAccessor<T>>(matrix: &Matrix<T, E>) -> usize {
    unimplemented!();
}

/// Computes $A^{-1}$ from matrix $A$ using the Gauss-Jordan method. It runs in $O(n^3)$. This
/// implementation does not provide any guarantees about its numerical stability.
///
/// * `matrix` - the matrix to invert, corresponds to $A$ above
pub fn inverse_naive<T: Field<T>, E: MatrixAccessor<T>>(matrix: &Matrix<T, E>) -> Matrix<T, E> {
    let mut augmented = matrix.elements();
    let size = matrix.width();
    let identity = Matrix::identity(size);
    for row in 0..size {
        for col in 0..size {
            augmented[row].push(identity[row][col]);
        }
    }
    let mut augmented = Matrix::new(augmented);
    let mut reduced = augmented.reduce().borrow().elements();
    for i in 0..size {
        for j in size..2 * size {
            reduced[i][j] = reduced[i][j] / reduced[i][i];
        }
    }
    let mut inverse = Vec::new();
    for i in 0..size {
        inverse.push(reduced[i][size..].to_vec());
    }
    Matrix::new(inverse)
}

pub fn gram_schmidt<T: Field<T>, E: MatrixAccessor<T>>(matrix: &Matrix<T, E>) -> Matrix<T, E> {
    unimplemented!();
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    const MATRIX_BENCH_SIZE: usize = 200;

    #[test]
    fn test_mat_mul_naive() {
        let lhs = Matrix::<u32>::identity(5);
        let rhs = Matrix::new(vec![vec![1, 2, 3, 4, 5]; 5]);
        assert_eq!(
            mat_mul_naive(&lhs, &rhs),
            Matrix::new(vec![vec![1, 2, 3, 4, 5]; 5])
        );
    }

    #[bench]
    fn bench_mat_mul_naive(b: &mut Bencher) {
        b.iter(|| {
            mat_mul_naive(
                &Matrix::new(vec![vec![1; MATRIX_BENCH_SIZE]; MATRIX_BENCH_SIZE]),
                &Matrix::new(vec![vec![2; MATRIX_BENCH_SIZE]; MATRIX_BENCH_SIZE]),
            )
        });
    }

    #[test]
    fn test_mat_mul_naive_threaded() {
        let lhs = Matrix::<u32>::identity(5);
        let rhs = Matrix::new(vec![vec![1, 2, 3, 4, 5]; 5]);
        assert_eq!(
            mat_mul_naive_threaded(&lhs, &rhs),
            Matrix::new(vec![vec![1, 2, 3, 4, 5]; 5])
        );
    }

    #[bench]
    fn bench_mat_mul_naive_threaded(b: &mut Bencher) {
        b.iter(|| {
            mat_mul_naive_threaded(
                &Matrix::new(vec![vec![1; MATRIX_BENCH_SIZE]; MATRIX_BENCH_SIZE]),
                &Matrix::new(vec![vec![2; MATRIX_BENCH_SIZE]; MATRIX_BENCH_SIZE]),
            )
        });
    }

    #[test]
    fn test_gauss_elim_naive() {
        let matrix = Matrix::new(vec![vec![2.0, -1.0, 1.0], vec![1.0, 1.0, 5.0]]);
        assert_eq!(
            gauss_elim_naive(&matrix),
            Matrix::new(vec![vec![2.0, -1.0, 1.0], vec![0.0, 3.0, 9.0]])
        );
    }

    #[bench]
    fn bench_gauss_elim_naive(b: &mut Bencher) {
        b.iter(|| {
            gauss_elim_naive(&Matrix::new(vec![
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                vec![31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                vec![41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                vec![51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                vec![61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                vec![71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                vec![81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
                vec![91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
            ]))
        })
    }

    #[test]
    fn test_inverse_naive() {
        let matrix = Matrix::<u32>::identity(5);
        assert_eq!(inverse_naive(&matrix), Matrix::<u32>::identity(5));
        let matrix = Matrix::new(vec![vec![1, 0, 0], vec![0, 1, 0], vec![1, 0, 1]]);
        assert_eq!(
            inverse_naive(&matrix),
            Matrix::new(vec![vec![1, 0, 0], vec![0, 1, 0], vec![-1, 0, 1]])
        );
    }

    #[test]
    #[ignore]
    fn det_square_matrix() {
        let matrix = Matrix::<i32>::new(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(det_naive(&matrix), -2);
        let matrix = Matrix::<i32>::new(vec![
            vec![4, 3, 2, 1],
            vec![1, 2, 3, 4],
            vec![3, 4, 1, 2],
            vec![2, 1, 4, 4],
        ]);
        assert_eq!(det_naive(&matrix), 20);
    }

    #[test]
    #[ignore]
    fn euclidean_scalar_product() {
        unimplemented!();
    }

    #[test]
    #[ignore]
    fn euclidean_norm() {
        unimplemented!();
    }
}
