#![allow(non_snake_case)]

extern crate rand;
extern crate nalgebra as na;
extern crate lapack;
extern crate hf;

use lapack::fortran::*;
use na::{Matrix, MatrixN, VectorN, ArrayStorage, U7, SymmetricEigen};

use hf::integrals::{Integrals, get_integrals};

type Mat77 = Matrix<f64, U7, U7, ArrayStorage<f64, U7, U7>>;
type Vec7 = VectorN<f64, U7>;

fn main() {
    let N = 6 + 1 + 1; // 8
    let ints = get_integrals("h2oints.txt").unwrap();
    let S = Mat77::from_iterator(ints.overlap.iter().flatten().map(|i| i.clone()));
    let T = Mat77::from_iterator(ints.kinetic.iter().flatten().map(|i| i.clone()));
    let V = Mat77::from_iterator(ints.potential.iter().flatten().map(|i| i.clone()));
    let H = T + V;

    let (U_S, L_S) = diag(S);
    let rootL_S = L_S.map(|x| if x > 10.0e-6 {x.powf(-0.5)} else {0f64});
    let rootS = U_S * rootL_S * U_S.transpose();

    let F0 = rootS.transpose() * H * rootS;
    let (U_F0, _) = diag(F0);
    let mut C0 = rootS * U_F0;
    let mut E_ref = 0.0;

    let n_iter = 200;
    for i in 0..n_iter {
        let (E, C, F, D, _epsilon) = scf(C0, &H, &rootS, N, &ints);
        println!("{} {}", i, E);

        if test_converge(E, E_ref, &D, &F, &S) {
            println!("{}", C);
            return;
        }

        E_ref = E;
        C0 = C;
    }

    println!("did not converge in {} iterations.", n_iter)
}

fn scf(C: Mat77, H: &Mat77, rootS: &Mat77, N: usize, ints: &Integrals) -> (f64, Mat77, Mat77, Mat77, Mat77) {
    let mut D = Mat77::zeros();
    for m in 0..7 {
        for n in 0..7 {
            for i in 0..N/2 + 1 {
                D[m+n*7] += C[m+i*7] * C[n+i*7];
            }
        }
    }

    let mut F = H.clone();
    for m in 0..7 {
        for n in 0..7 {
            for r in 0..7 {
                for s in 0..7 {
                    F[m+n*7] += D[r+s*7] * {2.0*ints.two_e(m,n,r,s) - ints.two_e(m,r,n,s)}
                }
            }
        }
    }

    let mut E = ints.Enuc;
    for m in 0..7 {
        for n in 0..7 {
            E += D[m+n*7] * (H[m+n*7] + F[m+n*7])
        }
    }

    let Fprime = rootS.transpose() * F * rootS;
    let (Cprime, epsilon) = diag(Fprime);

    return (E, rootS * Cprime, F, D, epsilon)
}

fn test_converge(E: f64, E_ref: f64, D: &Mat77, F: &Mat77, S: &Mat77) -> bool {
        if E_ref - E < 10.0e-9 {
            let mut DF = Mat77::zeros();
            for m in 0..7 {
                for n in 0..7 {
                    for r in 0..7 {
                        for s in 0..7 {
                            DF[m+n*7] += S[m+r*7]*D[r+s*7]*F[s+n*7] - S[s+n*7]*D[r+s*7]*F[m+r*7]
                        }
                    }
                }
            }

            let mut norm = 0.0f64;
            for i in 0..7 {
                for     j in 0..7 {
                    norm += DF[i+j*7].powi(2);
                }
            }
            norm = norm.powf(0.5);

            if norm < 10e-9 {
                return true;
            }
        }

    return false;
}

fn diag(mut A: Mat77) -> (MatrixN<f64, U7>, MatrixN<f64, U7>) {
    let mut w = Vec7::from_element(0f64); // eigenvalue output
    let mut work = vec![0f64; 49];
    let lwork = work.len() as i32;
    let mut info = 0;

    dsyev(b'V', b'U', 7, &mut A.as_mut_slice(), 7, w.as_mut(), &mut work, lwork, &mut info);

    return (A, Mat77::from_diagonal(&w))
}

// fn diag(mut A: Mat77) -> (MatrixN<f64, U7>, MatrixN<f64, U7>) {
//     let A_diag = SymmetricEigen::new(A);
//     let mut U = A_diag.eigenvectors;
//     let mut V = A_diag.eigenvalues;
//
//     // bubble sort
//     let mut elementsToCheck = V.len();
//     let mut swaps = 1;
//     while swaps > 0 {
//         swaps = 0;
//         for i in 0..elementsToCheck - 1 {
//             if V[i] > V[i+1] {
//                 // swap these values
//                 let tmp = V[i+1];
//                 V[i+1] = V[i];
//                 V[i] = tmp;
//
//                 U.swap_columns(i,i+1);
//
//                 swaps += 1;
//             }
//         }
//
//         elementsToCheck -= 1;
//     }
//
//     return (U, Mat77::from_diagonal(&V));
// }

#[cfg(test)]
mod tests {
    use super::*;
    use hf::*;

    #[test]
    fn diag_works() {
        let mut A = Mat77::new_random();
        // make A symmetric
        for i in 0..7 {
            for j in 0..7 {
                A[j*7 + i] = A[i*7 + j]
            }
        }

        let (P,D) = diag(A.clone());
        let B = P*D*P.transpose();
        for i in 0..49 {
            assert!(A[i] - B[i] < 0.000001, "{} {}", A[i], B[i]);
        }
    }

    #[test]
    fn two_e_permutations() {
        // should obey 8-fold symmetry
        // [pq|rs] = [qp|rs] = [pq|sr] = [qp|sr] = [rs|pq] = [sr|pq] = [rs|qp] = [sr|qp]
        let ints = get_integrals("h2oints.txt").unwrap();
        for p in 0..7 {
            for q in 0..7 {
                for r in 0..7 {
                    for s in 0..7 {
                        let x = ints.two_e(p,q,r,s);
                        assert_eq!(x, ints.two_e(q, p, r, s),
                            "{} {} {} {}", p, q, r, s);
                        assert_eq!(x, ints.two_e(q, p, s, r),
                            "{} {} {} {}", p, q, r, s);
                        assert_eq!(x, ints.two_e(p, q, s, r),
                            "{} {} {} {}", p, q, r, s);
                        assert_eq!(x, ints.two_e(r, s, p, q),
                            "{} {} {} {}", p, q, r, s);
                        assert_eq!(x, ints.two_e(r, s, q, p),
                            "{} {} {} {}", p, q, r, s);
                        assert_eq!(x, ints.two_e(s, r, p, q),
                            "{} {} {} {}", p, q, r, s);
                        assert_eq!(x, ints.two_e(s, r, q, p),
                            "{} {} {} {}", p, q, r, s);
                    }
                }
            }
        }
    }
}
