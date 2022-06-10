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

use crate::algebra::matrix::*;

/// A straightforward implementation of the euclidean scalar product
/// $\langle x, y \rangle = x^T y$ of two vectors $x, y$ that returns a scalar matrix (a 1 x 1 matrix) corresponding to the
/// numerical value of the scalar product.
///
/// * `lhs` - corresponds to $x^T$ above, is expected to be in transposed form ($x^T$) already,
/// thus of size $1 \times d$
/// * `rhs` - corresponds to $y$ above, expected to be of size $d \times 1$
pub fn euclidean_scalar_product_naive<T: Field<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
    let mut value = lhs[0][0] * rhs[0][0];
    for i in 1..lhs.width() {
        value = value + lhs[0][i] * rhs[i][0];
    }
    Matrix::scalar(value)
}

/// A naive implementation of the euclidean norm (colloquially, the 'length')
/// $\lvert \lvert x \rvert \rvert = \sqrt{\langle x, x \rangle}$ of a vector $x$ that returns a
/// scalar matrix (a 1 x 1 matrix) corresponding to the numerical value of the norm.
///
/// * `matrix` - corresponds to $x$ above, expected to be of size $\dim x \times 1$
pub fn euclidean_norm_naive<T: Field<T> + Float>(matrix: &Matrix<T>) -> Matrix<T> {
    Matrix::scalar(
        euclidean_scalar_product_naive(matrix, matrix)
            .to_scalar()
            .sqrt(),
    )
}

/// A naive implementation of the more general $p$-norm: $\lvert \lvert x \rvert \rvert_p =
/// (\lvert x_1 \rvert ^ p + ... + \lvert x_n \rvert ^ p)^\frac{1}{p}$.
///
/// * `matrix` - corresponds to the $x$ above, expected to be of size $\dim x \times 1$
pub fn p_norm_naive<T: Field<T> + Float + Signed>(matrix: &Matrix<T>, p: usize) -> Matrix<f64> {
    let mut value: T = num::zero();
    for i in 0..matrix.height() {
        value = value + pow(abs(matrix[i][0]), p);
    }
    Matrix::<f64>::scalar(nth_root(value.to_f64().unwrap(), p as f64))
}

/// [https://rosettacode.org/wiki/Nth_root#Rust](https://rosettacode.org/wiki/Nth_root#Rust)
///
/// Solves for $x$ in $x^n = v$.
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
/// returns a matrix of size $n \times p$.
///
/// * `lhs` - corresponds to $A$ above, expected to be of size $n \times m$
/// * `rhs` - corresponds to $B$ above, expected to be of size $m \times p$
pub fn mat_mul_naive<T: Field<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
    let mut elements = Vec::new();
    for row in 0..lhs.height() {
        elements.push(Vec::new());
        for col in 0..rhs.width() {
            let mut r = Vec::new();
            for i in 0..rhs.height() {
                r.push(rhs[i][col]);
            }
            elements[row].push(
                (Matrix::vector(lhs[row].to_vec()).transpose() * Matrix::<T>::vector(r))
                    .to_scalar(),
            );
        }
    }
    Matrix::new(elements)
}

/// A classic, naive implementation of the gaussian row reduction algorithm, with runtime
/// complexity $O(n^3)$. Returns a new matrix that corresponds to the row-reduced variant of the
/// input matrix. Transforms the input matrix into row-echelon form.
///
/// * `matrix` - the matrix to reduce
pub fn gauss_elim_naive<T: Field<T>>(matrix: &Matrix<T>) -> Matrix<T> {
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

pub fn lu_decomp_naive<T: Field<T>>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    unimplemented!();
}

pub fn qr_decomp_naive<T: Field<T>>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    unimplemented!();
}

pub fn det_naive<T: Field<T>>(matrix: &Matrix<T>) -> T {
    unimplemented!();
}

pub fn rank_naive<T: Field<T>>(matrix: &Matrix<T>) -> usize {
    unimplemented!();
}

pub fn inverse_naive<T: Field<T>>(matrix: &Matrix<T>) -> Matrix<T> {
    unimplemented!()
}

pub fn gram_schmidt<T: Field<T>>(matrix: &Matrix<T>) -> Matrix<T> {
    unimplemented!();
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[test]
    fn mat_mul() {
        let lhs = Matrix::<u32>::identity(5);
        let rhs = Matrix::new(vec![vec![1, 2, 3, 4, 5]; 5]);
        assert_eq!(lhs * rhs, Matrix::new(vec![vec![1, 2, 3, 4, 5]; 5]));
    }

    #[bench]
    fn bench_mat_mul_10_by_10(b: &mut Bencher) {
        b.iter(|| {
            mat_mul_naive(
                &Matrix::new(vec![vec![1; 10]; 10]),
                &Matrix::new(vec![vec![2; 10]; 10]),
            )
        });
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
