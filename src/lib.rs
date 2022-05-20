use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, PartialEq, Eq)]
pub struct Matrix<T> {
    elements: Vec<Vec<T>>,
    width: usize,
    height: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Vector<T> {
    elements: Vec<T>,
    dimension: usize,
}

impl<T: Clone + Copy> Matrix<T> {
    pub fn new(elements: &Vec<Vec<T>>) -> Matrix<T> {
        if elements.is_empty() {
            panic!("attempting to create matrix with no elements");
        }
        let height = elements.len();
        let width = elements[0].len();
        for row in elements {
            if row.len() != width {
                panic!(
                    "matrix rows do not contain equal number of elements, expected {}, got {}",
                    width,
                    row.len()
                );
            }
        }
        Matrix {
            elements: elements.to_vec(),
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

    fn assert_can_multiply(&self, rhs: &Self) {
        if self.width != rhs.height {
            panic!(
                "matrices must be of sizes n x m and m x p to multiply, got {} x {} and {} x {}",
                self.height, self.width, rhs.height, rhs.width
            );
        }
    }
}

impl<T: Add<T, Output = T> + AddAssign<T> + Mul<T, Output = T> + Clone + Copy> Vector<T> {
    fn new(elements: &Vec<T>) -> Vector<T> {
        if elements.is_empty() {
            panic!("vector has dimension 0");
        }
        Vector {
            elements: elements.to_vec(),
            dimension: elements.len(),
        }
    }

    fn assert_same_dimension(&self, rhs: &Vector<T>) {
        if self.dimension != rhs.dimension {
            panic!(
                "vector must be of same dimension to compute scalar product, got {} and {}",
                self.dimension, rhs.dimension
            );
        }
    }
}

impl<T: Add<T, Output = T> + AddAssign<T> + Mul<T, Output = T> + Clone + Copy> Mul for Vector<T> {
    type Output = T;

    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_same_dimension(&rhs);
        let mut value: T = self[0] * rhs[0];
        for i in 1..self.dimension {
            value += self[i] * rhs[i];
        }
        value
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = Vec<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl<T: Add<T, Output = E> + Copy, E: Clone + Copy> Add for Matrix<T> {
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
        Matrix::<E>::new(&elements)
    }
}

impl<T: Neg<Output = E> + Copy, E: Clone + Copy> Neg for Matrix<T> {
    type Output = Matrix<E>;

    fn neg(self) -> Self::Output {
        let mut elements = Vec::<Vec<E>>::new();
        for row in 0..self.height {
            elements.push(Vec::<E>::new());
            for col in 0..self.width {
                elements[row].push(-self[row][col]);
            }
        }
        Matrix::<E>::new(&elements)
    }
}

impl<T: Add<T, Output = T> + AddAssign<T> + Mul<T, Output = T> + Clone + Copy> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_can_multiply(&rhs);
        let mut elements = Vec::<Vec<T>>::new();
        for row in 0..self.height {
            elements.push(Vec::<T>::new());
            for col in 0..rhs.width {
                elements[row].push(Vector::<T>::new(&self[row]) * Vector::<T>::new(&rhs[col]));
            }
        }
        Matrix::<T>::new(&elements)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn matrix_multiplication() {
        let left = Matrix::<u32>::new(&vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let right = Matrix::<u32>::new(&vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(
            left * right,
            Matrix::<u32>::new(&vec![vec![9, 6, 3], vec![54, 42, 30], vec![99, 78, 57]])
        );
    }

    #[test]
    fn scalar_product() {
        let left = Vector::<u32>::new(&vec![0, 1, 2]);
        let right = Vector::<u32>::new(&vec![3, 4, 5]);
        assert_eq!(left * right, 14);
    }

    #[test]
    #[should_panic]
    fn matrix_creation_inconsistent_rows() {
        let _ = Matrix::<u32>::new(&vec![vec![0, 1], vec![2, 3, 4, 5, 6]]);
    }

    #[test]
    fn matrix_indexing() {
        let matrix = Matrix::<u32>::new(&vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
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
        let matrix = Matrix::<u32>::new(&vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
        let _ = matrix[2][0];
    }

    #[test]
    #[should_panic]
    fn matrix_indexing_out_of_bounds_col() {
        let matrix = Matrix::<u32>::new(&vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
        let _ = matrix[1][5];
    }

    #[test]
    fn matrix_mut_indexing() {
        let mut matrix = Matrix::<u32>::new(&vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
        assert_eq!(
            matrix,
            Matrix::<u32>::new(&vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]])
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
            Matrix::<u32>::new(&vec![vec![9, 8, 7, 6, 5], vec![4, 3, 2, 1, 0]])
        );
    }

    #[test]
    fn matrix_negation() {
        let matrix = Matrix::<i32>::new(&vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        assert_eq!(
            -matrix,
            Matrix::<i32>::new(&vec![vec![0, -1, -2], vec![-3, -4, -5], vec![-6, -7, -8]])
        );
    }

    #[test]
    fn matrix_addition() {
        let left = Matrix::<u32>::new(&vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let right = Matrix::<u32>::new(&vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(left + right, Matrix::<u32>::new(&vec![vec![8, 8, 8]; 3]));
    }

    #[test]
    fn matrix_addition_with_negation() {
        let left = Matrix::<i32>::new(&vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
        let right = Matrix::<i32>::new(&vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(
            left + -right,
            Matrix::<i32>::new(&vec![vec![-8, -6, -4], vec![-2, 0, 2], vec![4, 6, 8]])
        );
    }

    #[test]
    #[should_panic]
    fn matrix_addition_incorrect_size() {
        let left = Matrix::<u32>::new(&vec![vec![0, 1, 2]]);
        let right = Matrix::<u32>::new(&vec![vec![3, 4, 5], vec![6, 7, 8]]);
        let _ = left + right;
    }
}
