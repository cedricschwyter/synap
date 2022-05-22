use super::algorithms::*;
use num::{Complex, Num, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

pub trait MatrixElement<T>:
    PartialEq + Debug + Copy + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + Zero + One
{
}

impl<T: Num + Debug + Copy> MatrixElement<T> for T {}

#[derive(Debug, PartialEq, Eq)]
pub struct Matrix<T: MatrixElement<T>> {
    elements: Vec<Vec<T>>,
    width: usize,
    height: usize,
    rank: Option<u32>,
    det: Option<T>,
    row_echelon_form: Option<Vec<Vec<T>>>,
    lu_decomposition: Option<Vec<Vec<T>>>,
    qr_decomposition: Option<Vec<Vec<T>>>,
    is_orthogonal: Option<bool>,
    is_normal: Option<bool>,
    is_orthonormal: Option<bool>,
    is_unitary: Option<bool>,
}

impl<T: MatrixElement<T>> Matrix<T> {
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
            row_echelon_form: None,
            lu_decomposition: None,
            qr_decomposition: None,
            is_normal: None,
            is_orthogonal: None,
            is_orthonormal: None,
            is_unitary: None,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

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

    pub fn identity_rect((height, width): (usize, usize)) -> Matrix<T> {
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

    pub fn vector(elements: Vec<T>) -> Matrix<T> {
        Matrix::<T>::new(vec![elements]).transpose()
    }

    pub fn scalar(element: T) -> Matrix<T> {
        Matrix::<T>::vector(vec![element])
    }

    pub fn to_scalar(&self) -> T {
        if self.width != 1 || self.height != 1 {
            panic!(
                "cannot convert matrix of size {} x {} to scalar, has to be 1 x 1",
                self.height, self.width
            );
        }
        self[0][0]
    }

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

    fn is_same_size(&self, rhs: &Self) -> bool {
        self.width == rhs.width && self.height == rhs.height
    }

    fn assert_same_size(&self, rhs: &Self) {
        if !self.is_same_size(rhs) {
            panic!(
                "matrices must be of equal size, got {} x {} and {} x {}",
                self.height, self.width, rhs.height, rhs.width
            );
        }
    }

    fn can_multiply(&self, rhs: &Self) -> bool {
        self.width == rhs.height
    }

    fn assert_can_multiply(&self, rhs: &Self) {
        if !self.can_multiply(rhs) {
            panic!(
                "matrices must be of sizes n x m and m x p to multiply, got {} x {} and {} x {}",
                self.height, self.width, rhs.height, rhs.width
            );
        }
    }

    fn is_scalar(&self) -> bool {
        self.width == 1 && self.height == 1
    }

    fn assert_scalar(&self) {
        if !self.is_scalar() {
            panic!(
                "expected scalar value, got {} x {}",
                self.height, self.width
            );
        }
    }

    fn is_square(&self) -> bool {
        self.width == self.height
    }

    fn assert_square(&self) {
        if !self.is_square() {
            panic!(
                "expected square matrix, got {} x {}",
                self.height, self.width
            );
        }
    }

    fn det(&mut self) -> T {
        if let Some(det) = self.det {
            return det;
        }
        let det = det_naive(self);
        self.det = Some(det);
        det
    }

    fn rank(&mut self) -> u32 {
        if let Some(rank) = self.rank {
            return rank;
        }
        let rank = rank_naive(self);
        self.rank = Some(rank);
        rank
    }

    fn is_singular(&mut self) -> bool {
        self.det();
        if let Some(det) = self.det {
            return det == num::zero();
        }
        false
    }

    fn assert_singular(&mut self) {
        if !self.is_singular() {
            panic!("expected matrix to be singular, but determinant is not 0",);
        }
    }

    fn is_regular(&mut self) -> bool {
        !self.is_singular()
    }

    fn assert_regular(&mut self) {
        if !self.is_regular() {
            panic!("expected matrix to be regular, but determinant is 0");
        }
    }
}

impl<T: MatrixElement<T> + Num> Matrix<Complex<T>> {
    pub fn hermitian(&self) -> Matrix<Complex<T>> {
        unimplemented!();
    }
}

impl<T: MatrixElement<T>> Index<usize> for Matrix<T> {
    type Output = Vec<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T: MatrixElement<T>> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl<T: MatrixElement<T>> Add for Matrix<T> {
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

impl<T: MatrixElement<T>> Sub for Matrix<T> {
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

impl<T: MatrixElement<T> + Neg<Output = T>> Neg for Matrix<T> {
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

impl<T: MatrixElement<T>> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_can_multiply(&rhs);
        if self.height == 1 && rhs.width == 1 {
            return vec_vec_mul_naive(&self, &rhs);
        }
        mat_mat_mul_naive(&self, &rhs)
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
        let identity = Matrix::<i32>::identity_rect((4, 4));
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
    #[ignore]
    fn hermitian_transpose() {
        let matrix = Matrix::<Complex<i32>>::new(vec![vec![]]);
        assert_eq!(
            matrix.transpose(),
            Matrix::<Complex<i32>>::new(vec![vec![]])
        );
    }
}
