use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

// waiting for the compiler to implement this feature, until then manually enforcing these trait bounds,
// but already preparing the type for easy switch -- see https://github.com/rust-lang/rust/issues/21903
//type MatrixElement<T: Add<T, Output = T> + Mul<T, Output = T> + Copy> = T;
pub type MatrixElement<T> = T;

#[derive(Debug, PartialEq, Eq)]
pub struct Matrix<T> {
    elements: Vec<Vec<MatrixElement<T>>>,
    width: usize,
    height: usize,
}

impl<T> Matrix<MatrixElement<T>> {
    pub fn new<E>(elements: Vec<Vec<MatrixElement<E>>>) -> Matrix<MatrixElement<E>> {
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
        }
    }
    fn assert_same_size(&self, rhs: &Self) {
        if self.width != rhs.width || self.height != rhs.height {
            panic!(
                "matrices must be of equal size, got {} x {} and {} x {}",
                self.height, self.width, rhs.height, rhs.width
            );
        }
    }
}

impl<T> Index<usize> for Matrix<MatrixElement<T>> {
    type Output = Vec<MatrixElement<T>>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T: Add<Output = T> + Copy> Add for Matrix<MatrixElement<T>> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.assert_same_size(&rhs);
        let mut entries = Vec::<Vec<MatrixElement<T>>>::new();
        for row in 0..self.height {
            entries.push(Vec::<MatrixElement<T>>::new());
            for col in 0..self.width {
                entries[row].push(self[row][col] + rhs[row][col]);
            }
        }
        Matrix {
            elements: entries,
            width: self.width,
            height: self.height,
        }
    }
}

impl<T> Mul for Matrix<MatrixElement<T>> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Matrix {
            elements: self.elements,
            width: self.width,
            height: self.height,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn matrix_addition() {
        let left = Matrix::<u32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let right = Matrix::<u32>::new(vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(left + right, Matrix::<u32>::new(vec![vec![8, 8, 8]; 3]));
    }
}
