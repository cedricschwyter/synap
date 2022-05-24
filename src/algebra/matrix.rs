//! This module defines one of the most fundamental objects of the project: the matrix.
//! Be careful, it is also the only abstract mathematical object that will be implemented, as
//! vectors and scalars are also just matrices of special sizes in the end. This simplifies a lot
//! of the functionality of the project and reduces a bunch of boilerplate code.

use super::algorithms::*;
use num::{Complex, Num, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Rem, Sub};

/// A trait to ensure that matrix elements support the most basic of operations, as otherwise the
/// matrix implementation is quite literally useless.
/// Note that this trait essentially defines the axioms of a field. Therefore matrices can be
/// constructed over an arbitrary field and are not restricted to the built-in numeric/complex
/// types, and it is guaranteed that all the algorithms work. Except for the additive inverse
/// operation (trait [`Neg`](std::ops::Neg)) all field axioms are enforced by the compiler. We deliberately do not
/// require the [`Neg`](std::ops::Neg) trait to be implemented, as this would restrict the type `T` to only signed
/// types, which may not be required in all situations. In functions/methods where the additive
/// inverse operation is required it is bounded separately.
pub trait FieldElement<T>:
    PartialEq
    + Debug
    + Copy
    + Add<T, Output = T>
    + Sub<T, Output = T>
    + Mul<T, Output = T>
    + Div<T, Output = T>
    + Rem<T, Output = T>
    + Zero
    + One
{
}

/// Blanket-implementation for all built-in numeric types.
impl<T: Num + Debug + Copy> FieldElement<T> for T {}

/// The matrix. The fundamental building block of this crate. A very versatile struct, intending to
/// perform expensive computations only once and caching the results. The struct is guaranteed to be
/// in an internally consistent state at all times.
#[derive(Debug, PartialEq, Eq)]
pub struct Matrix<T: FieldElement<T>> {
    elements: Vec<Vec<T>>,
    width: usize,
    height: usize,
    rank: Option<usize>,
    det: Option<T>,
    inverse: Option<Box<Matrix<T>>>,
    row_echelon_form: Option<Box<Matrix<T>>>,
    lu_decomposition: Option<(Box<Matrix<T>>, Box<Matrix<T>>)>,
    qr_decomposition: Option<(Box<Matrix<T>>, Box<Matrix<T>>)>,
    is_orthogonal: Option<bool>,
    is_normal: Option<bool>,
    is_orthonormal: Option<bool>,
    is_unitary: Option<bool>,
}

/// Default implementation of functions and methods for arbitrary fields.
impl<T: FieldElement<T>> Matrix<T> {
    /// Default constructor. All matrix initialization is supposed to go through this call to
    /// ensure internal consistency with dimension values.
    ///
    /// * `elements` - a two dimensional vector of field elements representing the rows of a
    /// matrix, rows are expected to be of the same length
    pub fn new(elements: Vec<Vec<T>>) -> Matrix<T> {
        if elements.is_empty() {
            panic!("attempting to create matrix with no elements");
        }
        let height = elements.len();
        let width = elements[0].len();
        for row in &elements {
            if row.len() != width {
                panic!(
                    "matrix rows do not contain equal number of elements, expected {}, got {}",
                    width,
                    row.len()
                );
            }
        }
        Matrix {
            elements,
            width,
            height,
            rank: None,
            det: None,
            inverse: None,
            row_echelon_form: None,
            lu_decomposition: None,
            qr_decomposition: None,
            is_normal: None,
            is_orthogonal: None,
            is_orthonormal: None,
            is_unitary: None,
        }
    }

    /// Getter for the width of the matrix
    pub fn width(&self) -> usize {
        self.width
    }

    /// Getter for the height of the matrix
    pub fn height(&self) -> usize {
        self.height
    }

    /// Creates an identity matrix of square dimensions as given by the parameter.
    /// Returns
    /// $$
    /// \mathbb{I}\_d
    /// = \begin{bmatrix}
    /// 1 & 0 & ... & 0 \\\\
    /// 0 & 1 & ... & 0 \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// 0 & 0 & ... & 1
    /// \end{bmatrix}
    /// $$
    /// where $d$ is the dimension.
    ///
    /// * `dimension` - the dimension of the square identity matrix, given above by $d$
    pub fn identity(dimension: usize) -> Matrix<T> {
        let mut elements = vec![vec![num::zero(); dimension]; dimension];
        for row in 0..dimension {
            for col in 0..dimension {
                if row == col {
                    elements[row][col] = num::one();
                }
            }
        }
        Matrix::<T>::new(elements)
    }

    /// Creates an identity matrix of rectangular dimensions as given by the parameters.
    /// Returns
    /// $$
    /// \mathbb{I}\_{n \times m}
    /// = \begin{bmatrix}
    /// 1 & 0 & ... & 0 & ... \\\\
    /// 0 & 1 & ... & 0 & ... \\\\
    /// \vdots & \vdots & \ddots & \vdots & ... \\\\
    /// 0 & 0 & ... & 1 & ... \\\\
    /// \vdots & \vdots & \vdots & \vdots & \ddots
    /// \end{bmatrix}
    /// $$
    /// where $n \times m$ is the dimension.
    ///
    /// * `height` - the height of the identity matrix, given above by $n$
    /// * `width` - the width of the identity matrix, given above by $m$
    pub fn identity_rect(height: usize, width: usize) -> Matrix<T> {
        let mut elements = vec![vec![num::zero(); width]; height];
        for row in 0..height {
            for col in 0..width {
                if row == col {
                    elements[row][col] = num::one();
                }
            }
        }
        Matrix::<T>::new(elements)
    }

    /// Convenience constructor for a vector $x$, that is, a $\dim x \times 1$ matrix.
    ///
    /// * `elements` - a vector of length $\dim x$ of elements corresponding to the elements of $x$
    pub fn vector(elements: Vec<T>) -> Matrix<T> {
        Matrix::<T>::new(vec![elements]).transpose()
    }

    /// Convenience constructor for a scalar $\alpha$, that is, a $1 \times 1$ matrix.
    ///
    /// * `element` - a single element representing the scalar value of $\alpha$
    pub fn scalar(element: T) -> Matrix<T> {
        Matrix::<T>::vector(vec![element])
    }

    /// Unwraps the $1 \times 1$ matrix into the underlying field element $\alpha$.
    /// Panics if the matrix is not of dimension $1 \times 1$.
    pub fn to_scalar(&self) -> T {
        self.assert_scalar();
        self[0][0]
    }

    /// Computes the transpose of the matrix, that is, if the matrix $A$ is of dimension $n \times
    /// m$ the method returns $A^T$ of size $m \times n$.
    pub fn transpose(&self) -> Matrix<T> {
        let mut elements = Vec::<Vec<T>>::new();
        for col in 0..self.width {
            elements.push(Vec::<T>::new());
            for row in 0..self.height {
                elements[col].push(self[row][col]);
            }
        }
        Matrix::<T>::new(elements)
    }

    /// Scales a matrix $A$ by a scalar $\alpha$.
    ///
    /// * `lhs` - corresponds to $\alpha$ above, expected to be of size $1 \times 1$
    pub fn scale(&self, lhs: Matrix<T>) -> Matrix<T> {
        lhs.assert_scalar();
        let mut elements = Vec::<Vec<T>>::new();
        for row in 0..self.height {
            elements.push(Vec::<T>::new());
            for col in 0..self.width {
                elements[row].push(lhs.to_scalar() * self[row][col]);
            }
        }
        Matrix::<T>::new(elements)
    }

    /// Computes the inverse of the matrix $A$, that is, returns $A^{-1}$ if it exists. Panics
    /// otherwise.
    ///
    /// ## Caution:
    /// This method can incur unexpected comparatively expensive computations if $A$ is a square matrix and $\det A$, $\text{rank} A$ and $A^{-1}$ have not
    /// already been computed and cached.
    pub fn inverse(mut self) -> Matrix<T> {
        if let Some(inverse) = self.inverse {
            return *inverse;
        }
        self.assert_regular();
        let inverse = inverse_naive(&self);
        self.inverse = Some(Box::new(inverse));
        if let Some(inverse) = self.inverse {
            return *inverse;
        }
        panic!("inconsistent internal state of matrix");
    }

    /// Checks whether two matrices of the same type are the same size. This is exactly then the
    /// case for two matrices $A$ and $B$ when $A$ and $B$ are both of dimension $n \times m$.
    ///
    /// * `rhs` - the matrix to compare against, corresponds to $B$ above
    pub fn is_same_size(&self, rhs: &Self) -> bool {
        self.width == rhs.width && self.height == rhs.height
    }

    /// Asserts that two matrices of the same type are the same size. This is exactly then the
    /// case for two matrices $A$ and $B$ when $A$ and $B$ are both of dimension $n \times m$.
    /// Panics otherwise.
    ///
    /// * `rhs` - the matrix to compare against, corresponds to $B$ above
    pub fn assert_same_size(&self, rhs: &Self) {
        if !self.is_same_size(rhs) {
            panic!(
                "matrices must be of equal size, got {} x {} and {} x {}",
                self.height, self.width, rhs.height, rhs.width
            );
        }
    }

    /// Checks whether two matrices can be multiplied. This is exactly then the case for two
    /// matrices $A$ and $B$ when $A$ is of dimension $n \times m$ and $B$ is of dimension $m
    /// \times p$.
    ///
    /// * `rhs` - the right hand side of the multiplication, corresponds to $B$ above
    pub fn can_multiply(&self, rhs: &Self) -> bool {
        self.width == rhs.height
    }

    /// Asserts that two matrices can be multiplied. This is exactly then the case for two
    /// matrices $A$ and $B$ when $A$ is of dimension $n \times m$ and $B$ is of dimension $m
    /// \times p$. Panics otherwise.
    ///
    /// * `rhs` - the right hand side of the multiplication, corresponds to $B$ above
    pub fn assert_can_multiply(&self, rhs: &Self) {
        if !self.can_multiply(rhs) {
            panic!(
                "matrices must be of sizes n x m and m x p to multiply, got {} x {} and {} x {}",
                self.height, self.width, rhs.height, rhs.width
            );
        }
    }

    /// Checks whether the matrix $x$ is a vector, that is, whether $x$ has dimension $\dim x \times 1$.
    pub fn is_vector(&self) -> bool {
        self.width == 1
    }

    /// Asserts that the matrix $x$ is a vector, that is, that $x$ has dimension $\dim x \times 1$. Panics otherwise.
    pub fn assert_vector(&self) {
        if !self.is_vector() {
            panic!("expected matrix to be 1 wide, but was {}", self.width);
        }
    }

    /// Checks whether the matrix $\alpha$ is a scalar, that is, whether $\alpha$ has dimension $1 \times 1$.
    pub fn is_scalar(&self) -> bool {
        self.width == 1 && self.height == 1
    }

    /// Asserts that the matrix $\alpha$ is a scalar, that is, that $\alpha$ has dimension $1 \times 1$. Panics otherwise.
    pub fn assert_scalar(&self) {
        if !self.is_scalar() {
            panic!(
                "expected scalar value, got {} x {}",
                self.height, self.width
            );
        }
    }

    /// Checks whether the matrix $A$ is a square matrix, that is, whether $A$ has dimension $n \times n$.
    pub fn is_square(&self) -> bool {
        self.width == self.height
    }

    /// Asserts that the matrix $A$ is a square matrix, that is, that $A$ has dimension $n \times n$. Panics otherwise.
    pub fn assert_square(&self) {
        if !self.is_square() {
            panic!(
                "expected square matrix, got {} x {}",
                self.height, self.width
            );
        }
    }

    /// Returns the determinant of the matrix $A$, that is, returns $\det A$.
    /// If $\det A$ has already been computed it returns the cached value. If not, and if $A$ is a
    /// square matrix, then it may perform the comparatively expensive computation. If $A$ is
    /// non-square then no determinant must be computed as then $\det A = 0$.
    ///
    /// ## Caution:
    /// This method can incur unexpected comparatively expensive computations if $A$ is a square matrix and both $\det A$ and $\text{rank} A$ have not
    /// already been computed and cached.
    ///
    /// ## Note:
    /// Here we made use of the provable fact that for any $n \times n$ matrix $A$ it holds that
    /// $$
    /// \det A \neq 0 \iff \text{rank} A = n.
    /// $$
    /// This allows to optimize the algorithm depending on whether the rank $\text{rank} A$ has
    /// already been computed.
    pub fn det(&mut self) -> T {
        if let Some(det) = self.det {
            return det;
        }
        let mut det = num::zero();
        if self.is_square() {
            if let Some(rank) = self.rank {
                if rank != self.height() {
                    self.det = Some(det);
                    return det;
                }
            }
            det = det_naive(self);
            self.det = Some(det);
            return det;
        }
        det = num::zero();
        self.det = Some(det);
        det
    }

    /// Returns the rank of the matrix $A$, that is, returns $\text{rank} A$.
    /// If $\text{rank} A$ has already been computed it returns the cached value, if not, then
    /// it may perform the comparatively expensive computation.
    ///
    /// ## Caution:
    /// This method can incur unexpected comparatively expensive computations if $A$ is a square matrix and both $\det A$ and $\text{rank} A$ have not
    /// already been computed and cached.
    ///
    /// ## Note:
    /// Here we made use of the provable fact that for any $n \times n$ matrix $A$ it holds that
    /// $$
    /// \det A \neq 0 \iff \text{rank} A = n.
    /// $$
    /// This allows to optimize the algorithm depending on whether the determinant $\det A$ has
    /// already been computed.
    pub fn rank(&mut self) -> usize {
        if let Some(rank) = self.rank {
            return rank;
        }
        if self.is_square() {
            if let Some(det) = self.det {
                if det != num::zero() {
                    let rank = self.height();
                    self.rank = Some(rank);
                    return rank;
                }
            }
        }
        let rank = rank_naive(self);
        self.rank = Some(rank);
        return rank;
    }

    /// Checks whether the matrix $A$ is full rank, i.e., if $A$ is of dimension $n \times n$ it checks whether $\text{rank} A = n$.
    /// Returns false otherwise as by the theorem we used in other methods already:
    ///
    /// ## Note:
    /// Here we made use of the provable fact that for any $n \times n$ matrix $A$ it holds that
    /// $$
    /// \det A \neq 0 \iff \text{rank} A = n.
    /// $$
    /// This allows to optimize the algorithm depending on whether the determinant $\det A$ or rank
    /// $\text{rank} A$ has already been computed.
    pub fn is_full_rank(&mut self) -> bool {
        if self.is_singular() {
            return false;
        }
        true
    }

    /// Asserts that the matrix $A$ is full rank, i.e., if $A$ is of dimension $n \times n$ it asserts that $\text{rank} A = n$.
    /// Panics otherwise as by the theorem we used in other methods already:
    ///
    /// ## Note:
    /// Here we made use of the provable fact that for any $n \times n$ matrix $A$ it holds that
    /// $$
    /// \det A \neq 0 \iff \text{rank} A = n.
    /// $$
    /// This allows to optimize the algorithm depending on whether the determinant $\det A$ or rank
    /// $\text{rank} A$ has already been computed.
    pub fn assert_full_rank(&mut self) {
        if !self.is_full_rank() {
            panic!("expected matrix to be full rank, but was not");
        }
    }

    /// Checks whether the matrix $A$ is singular, that is, whether its determinant $\det A$ is
    /// $\det A = 0$.
    ///
    /// ## Caution:
    /// This method can incur unexpected comparatively expensive computations if $A$ is a square matrix and both $\det A$ and $\text{rank} A$ have not
    /// already been computed and cached.
    pub fn is_singular(&mut self) -> bool {
        self.det() == num::zero()
    }

    /// Asserts that the matrix $A$ is singular, that is, that its determinant $\det A$ is
    /// $\det A = 0$. Panics otherwise.
    ///
    /// ## Caution:
    /// This method can incur unexpected comparatively expensive computations if $A$ is a square matrix and both $\det A$ and $\text{rank} A$ have not
    /// already been computed and cached.
    pub fn assert_singular(&mut self) {
        if !self.is_singular() {
            panic!("expected matrix to be singular, but determinant is not 0",);
        }
    }

    /// Checks whether the matrix $A$ is regular, that is, whether its determinant $\det A$ is
    /// $\det A \neq 0$.
    ///
    /// ## Caution:
    /// This method can incur unexpected comparatively expensive computations if $A$ is a square matrix and both $\det A$ and $\text{rank} A$ have not
    /// already been computed and cached.
    pub fn is_regular(&mut self) -> bool {
        !self.is_singular()
    }

    /// Asserts that the matrix $A$ is regular, that is, that its determinant $\det A$ is
    /// $\det A \neq 0$. Panics otherwise.
    ///
    /// ## Caution:
    /// This method can incur unexpected comparatively expensive computations if $A$ is a square matrix and both $\det A$ and $\text{rank} A$ have not
    /// already been computed and cached.
    pub fn assert_regular(&mut self) {
        if !self.is_regular() {
            panic!("expected matrix to be regular, but determinant is 0");
        }
    }

    /// Checks whether the matrix $A$ is symmetric, that is, whether it holds that $A^T = A$.
    pub fn is_symmetric(&self) -> bool {
        self.transpose() == *self
    }

    /// Asserts that the matrix $A$ is symmetric, that is, that it holds that $A^T = A$.
    /// Panics otherwise.
    pub fn assert_symmetric(&self) {
        if !self.is_symmetric() {
            panic!("expected matrix to be symmetic, but is not");
        }
    }

    /// Checks whether the matrix $A$ is skew-symmetric, that is, whether it holds that $A^T = -A$.
    pub fn is_skew_symmetric(&self) -> bool {
        unimplemented!();
    }

    /// Asserts that the matrix $A$ is skew-symmetric, that is, that it holds that $A^T = -A$.
    /// Panics otherwise.
    pub fn assert_skew_symmetric(&self) {
        if !self.is_skew_symmetric() {
            panic!("expected matrix to be skew-symmetric, but is not");
        }
    }
}

/// Defines an iterator for [`Matrix`](Matrix) structs
pub struct MatrixIterator<T: FieldElement<T>> {
    matrix: Matrix<T>,
    row: usize,
}

impl<T: FieldElement<T>> IntoIterator for Matrix<T> {
    type Item = Vec<T>;
    type IntoIter = MatrixIterator<T>;
    fn into_iter(self) -> Self::IntoIter {
        MatrixIterator {
            matrix: self,
            row: 0,
        }
    }
}

impl<T: FieldElement<T>> Iterator for MatrixIterator<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row < self.matrix.height() {
            let row = &self.matrix[self.row];
            self.row += 1;
            return Some(row.to_vec());
        }
        None
    }
}

/// Special implementation for complex numbers.
impl<T: FieldElement<T> + Num + Neg<Output = T>> Matrix<Complex<T>> {
    /// Computes the hermitian transpose of a matrix $A$, that is, computes $A^H$.
    pub fn hermitian(&self) -> Matrix<Complex<T>> {
        let mut transpose = self.transpose();
        for row in 0..transpose.height() {
            for col in 0..transpose.width() {
                transpose[row][col] = Complex {
                    re: transpose[row][col].re,
                    im: -transpose[row][col].im,
                };
            }
        }
        transpose
    }

    /// Checks whether the matrix $A$ is hermitian, that is, whether it holds that $A^H = A$.
    pub fn is_hermitian(&self) -> bool {
        self.hermitian() == *self
    }

    /// Asserts that the matrix $A$ is hermitian, that is, that it holds that $A^H = A$.
    pub fn assert_hermitian(&self) {
        if !self.is_hermitian() {
            panic!("expected matrix to be hermitian, but is not");
        }
    }
}

impl<T: FieldElement<T>> Index<usize> for Matrix<T> {
    type Output = Vec<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T: FieldElement<T>> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl<T: FieldElement<T>> Add for Matrix<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.assert_same_size(&rhs);
        let mut elements = Vec::<Vec<T>>::new();
        for row in 0..self.height {
            elements.push(Vec::<T>::new());
            for col in 0..self.width {
                elements[row].push(self[row][col] + rhs[row][col]);
            }
        }
        Matrix::<T>::new(elements)
    }
}

impl<T: FieldElement<T>> Sub for Matrix<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.assert_same_size(&rhs);
        let mut elements = Vec::<Vec<T>>::new();
        for row in 0..self.height {
            elements.push(Vec::<T>::new());
            for col in 0..self.width {
                elements[row].push(self[row][col] - rhs[row][col]);
            }
        }
        Matrix::<T>::new(elements)
    }
}

impl<T: FieldElement<T> + Neg<Output = T>> Neg for Matrix<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut elements = Vec::<Vec<T>>::new();
        for row in 0..self.height {
            elements.push(Vec::<T>::new());
            for col in 0..self.width {
                elements[row].push(-self[row][col]);
            }
        }
        Matrix::<T>::new(elements)
    }
}

impl<T: FieldElement<T>> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_can_multiply(&rhs);
        if self.height == 1 && rhs.width == 1 {
            return euclidean_scalar_product_naive(&self, &rhs);
        }
        mat_mul_naive(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_vector_multiplication() {
        let lhs = Matrix::<i32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let rhs = Matrix::<i32>::vector(vec![9, 10, 11]);
        assert_eq!(lhs * rhs, Matrix::<i32>::vector(vec![32, 122, 212]));
    }

    #[test]
    #[should_panic]
    fn matrix_vector_multiplication_incompatible_size() {
        let lhs = Matrix::<i32>::new(vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
        let rhs = Matrix::<i32>::vector(vec![6, 7, 8]);
        let _ = lhs * rhs;
    }

    #[test]
    fn matrix_multiplication() {
        let lhs = Matrix::<i32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let rhs = Matrix::<i32>::new(vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(
            lhs * rhs,
            Matrix::<i32>::new(vec![vec![9, 6, 3], vec![54, 42, 30], vec![99, 78, 57]])
        );
    }

    #[test]
    #[should_panic]
    fn matrix_multiplication_incompatible_size() {
        let lhs = Matrix::<i32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let rhs = Matrix::<i32>::new(vec![vec![8, 7, 6], vec![5, 4, 3]]);
        let _ = lhs * rhs;
    }

    #[test]
    fn scalar_product() {
        let lhs = Matrix::<u32>::vector(vec![0, 1, 2]).transpose();
        let rhs = Matrix::<u32>::vector(vec![3, 4, 5]);
        assert_eq!((lhs * rhs).to_scalar(), 14);
    }

    #[test]
    #[should_panic]
    fn matrix_creation_inconsistent_rows() {
        let _ = Matrix::<u32>::new(vec![vec![0, 1], vec![2, 3, 4, 5, 6]]);
    }

    #[test]
    fn matrix_indexing() {
        let matrix = Matrix::<u32>::new(vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
        assert_eq!(matrix[0][0], 0);
        assert_eq!(matrix[0][1], 1);
        assert_eq!(matrix[0][2], 2);
        assert_eq!(matrix[0][3], 3);
        assert_eq!(matrix[0][4], 4);
        assert_eq!(matrix[1][0], 5);
        assert_eq!(matrix[1][1], 6);
        assert_eq!(matrix[1][2], 7);
        assert_eq!(matrix[1][3], 8);
        assert_eq!(matrix[1][4], 9);
    }

    #[test]
    #[should_panic]
    fn matrix_indexing_out_of_bounds_row() {
        let matrix = Matrix::<u32>::new(vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
        let _ = matrix[2][0];
    }

    #[test]
    #[should_panic]
    fn matrix_indexing_out_of_bounds_col() {
        let matrix = Matrix::<u32>::new(vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
        let _ = matrix[1][5];
    }

    #[test]
    fn matrix_mut_indexing() {
        let mut matrix = Matrix::<u32>::new(vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
        assert_eq!(
            matrix,
            Matrix::<u32>::new(vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]])
        );
        matrix[0][0] = 9;
        matrix[0][1] = 8;
        matrix[0][2] = 7;
        matrix[0][3] = 6;
        matrix[0][4] = 5;
        matrix[1][0] = 4;
        matrix[1][1] = 3;
        matrix[1][2] = 2;
        matrix[1][3] = 1;
        matrix[1][4] = 0;
        assert_eq!(
            matrix,
            Matrix::<u32>::new(vec![vec![9, 8, 7, 6, 5], vec![4, 3, 2, 1, 0]])
        );
    }

    #[test]
    fn matrix_negation() {
        let matrix = Matrix::<i32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        assert_eq!(
            -matrix,
            Matrix::<i32>::new(vec![vec![0, -1, -2], vec![-3, -4, -5], vec![-6, -7, -8]])
        );
    }

    #[test]
    fn matrix_addition() {
        let lhs = Matrix::<u32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let rhs = Matrix::<u32>::new(vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(lhs + rhs, Matrix::<u32>::new(vec![vec![8, 8, 8]; 3]));
    }

    #[test]
    fn matrix_addition_with_negation() {
        let lhs = Matrix::<i32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let rhs = Matrix::<i32>::new(vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(
            lhs + -rhs,
            Matrix::<i32>::new(vec![vec![-8, -6, -4], vec![-2, 0, 2], vec![4, 6, 8]])
        );
    }

    #[test]
    #[should_panic]
    fn matrix_addition_incorrect_size() {
        let lhs = Matrix::<u32>::new(vec![vec![0, 1, 2]]);
        let rhs = Matrix::<u32>::new(vec![vec![3, 4, 5], vec![6, 7, 8]]);
        let _ = lhs + rhs;
    }

    #[test]
    fn scalar_matrix_multiplication() {
        let lhs = Matrix::<i32>::scalar(5);
        let rhs = Matrix::<i32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        assert_eq!(
            rhs.scale(lhs),
            Matrix::<i32>::new(vec![vec![0, 5, 10], vec![15, 20, 25], vec![30, 35, 40]])
        );
    }

    #[test]
    fn scalar_vector_multiplication() {
        let lhs = Matrix::<i32>::scalar(5);
        let rhs = Matrix::<i32>::vector(vec![0, 1, 2, 3, 4]);
        assert_eq!(
            rhs.scale(lhs),
            Matrix::<i32>::vector(vec![0, 5, 10, 15, 20])
        );
    }

    #[test]
    fn multi_step_computation() {
        let lhs = Matrix::<i32>::scalar(10);
        let middle_lhs = Matrix::<i32>::identity(4);
        let middle_rhs = Matrix::<i32>::vector(vec![1, 2, 3, 4]);
        let rhs = Matrix::<i32>::vector(vec![5, 6, 7, 8]);
        let identity = Matrix::<i32>::identity_rect(4, 4);
        assert_eq!(
            identity * (middle_lhs.scale(lhs) * -middle_rhs + rhs),
            Matrix::<i32>::vector(vec![-5, -14, -23, -32])
        );
    }

    #[test]
    fn identity_square() {
        let identity = Matrix::<i32>::identity(10);
        let matrix = Matrix::<i32>::new(vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 10]);
        assert_eq!(
            matrix * identity,
            Matrix::<i32>::new(vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 10])
        );
    }

    #[test]
    fn hermitian_transpose() {
        let matrix = Matrix::<Complex<i32>>::new(vec![
            vec![
                Complex::new(0, 0),
                Complex::new(0, 1),
                Complex::new(0, 2),
                Complex::new(0, 3),
                Complex::new(0, 4),
            ],
            vec![
                Complex::new(1, 0),
                Complex::new(1, 1),
                Complex::new(1, 2),
                Complex::new(1, 3),
                Complex::new(1, 4),
            ],
            vec![
                Complex::new(2, 0),
                Complex::new(2, 1),
                Complex::new(2, 2),
                Complex::new(2, 3),
                Complex::new(2, 4),
            ],
        ]);
        assert_eq!(
            matrix.hermitian(),
            Matrix::<Complex<i32>>::new(vec![
                vec![Complex::new(0, 0), Complex::new(1, 0), Complex::new(2, 0)],
                vec![
                    Complex::new(0, -1),
                    Complex::new(1, -1),
                    Complex::new(2, -1)
                ],
                vec![
                    Complex::new(0, -2),
                    Complex::new(1, -2),
                    Complex::new(2, -2)
                ],
                vec![
                    Complex::new(0, -3),
                    Complex::new(1, -3),
                    Complex::new(2, -3)
                ],
                vec![
                    Complex::new(0, -4),
                    Complex::new(1, -4),
                    Complex::new(2, -4)
                ]
            ])
        );
    }
}
