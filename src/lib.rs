use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, PartialEq, Eq)]
pub struct Matrix<T> {
    elements: Vec<Vec<T>>,
    width: usize,
    height: usize,
}

impl<T> Matrix<T> {
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

impl<T> Index<usize> for Matrix<T> {
    type Output = Vec<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T: Add<T, Output = E> + Copy, E> Add for Matrix<T> {
    type Output = Matrix<E>;

    fn add(self, rhs: Self) -> Self::Output {
        self.assert_same_size(&rhs);
        let mut elements = Vec::<Vec<E>>::new();
        for row in 0..self.height {
            elements.push(Vec::<E>::new());
            for col in 0..self.width {
                elements[row].push(self[row][col] + rhs[row][col]);
            }
        }
        Matrix::<E>::new(elements)
    }
}

impl<T: Neg<Output = E> + Copy, E> Neg for Matrix<T> {
    type Output = Matrix<E>;

    fn neg(self) -> Self::Output {
        let mut elements = Vec::<Vec<E>>::new();
        for row in 0..self.height {
            elements.push(Vec::<E>::new());
            for col in 0..self.width {
                elements[row].push(-self[row][col]);
            }
        }
        Matrix::<E>::new(elements)
    }
}

impl<T> Mul for Matrix<T> {
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
    #[should_panic]
    fn matrix_creation_inconsistent_rows() {
        let _ = Matrix::<u32>::new(vec![vec![0, 1], vec![2, 3, 4, 5, 6]]);
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
        let left = Matrix::<u32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let right = Matrix::<u32>::new(vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(left + right, Matrix::<u32>::new(vec![vec![8, 8, 8]; 3]));
    }

    #[test]
    fn matrix_addition_with_negation() {
        let left = Matrix::<i32>::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let right = Matrix::<i32>::new(vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(
            left + -right,
            Matrix::<i32>::new(vec![vec![-8, -6, -4], vec![-2, 0, 2], vec![4, 6, 8]])
        );
    }

    #[test]
    #[should_panic]
    fn matrix_addition_incorrect_size() {
        let left = Matrix::<u32>::new(vec![vec![0, 1, 2]]);
        let right = Matrix::<u32>::new(vec![vec![3, 4, 5], vec![6, 7, 8]]);
        let _ = left + right;
    }
}
