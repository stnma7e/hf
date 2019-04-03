extern crate regex;
extern crate lapack;
extern crate rand;
extern crate nalgebra as na;
extern crate nalgebra_lapack as nal;

use std::io;
use std::fs::File;
use std::io::prelude::*;
use regex::Regex;

use lapack::*;

use na::{Matrix, MatrixN, VectorN, ArrayStorage, U7, SymmetricEigen};

type Mat77 = Matrix<f64, U7, U7, ArrayStorage<f64, U7, U7>>;

#[derive(Default, Debug)]
struct Integrals {
    Enuc: f64,
    overlap: [[f64; 7]; 7],
    kinetic: [[f64; 7]; 7],
    potential: [[f64; 7]; 7],
    one_electron: [[f64; 7]; 7],
    two_electron: [[[[f64; 7]; 7]; 7]; 7]
}

impl Integrals {
    fn two_e(&self, mut u: usize, mut v: usize, mut p: usize, mut o: usize) -> f64 {
        /// Retuns two-electron integral for the given orbitals. Should obey 8-fold symmetry:
        /// [pq|rs] = [qp|rs] = [pq|sr] = [qp|sr] = [rs|pq] = [sr|pq] = [rs|qp] = [sr|qp].
        /// Where p >= q, and r >= s. And the integrals are ordered with pq orbitals first if
        /// (p(p + 1)/2) + q > (r(r + 1)/2) + s, otherwise the rs orbitals are first.

        // make sure u > v and p > o
        let mut tmp = 0;
        if v > u {
            tmp = u;
            u = v; v = tmp;
        }
        if o > p {
            tmp = p;
            p = o; o = tmp;
        }

        // make sure the order of uv and po is right
        let uv = (u*(u+1) / 2) + v;
        let po = (p*(p+1) / 2) + o;
        if uv >= po {
            self.two_electron[u][v][p][o]
        } else {
            self.two_electron[p][o][u][v]
        }
    }
}

fn main() {
    let N = 6 + 1 + 1; // 8
    let mut ints = get_integrals("h2oints.txt").unwrap();
    let S = Mat77::from_iterator(ints.overlap.iter().flatten().map(|i| i.clone()));
    let T = Mat77::from_iterator(ints.kinetic.iter().flatten().map(|i| i.clone()));
    let V = Mat77::from_iterator(ints.potential.iter().flatten().map(|i| i.clone()));
    let H = T + V;

    let (U_S, L_S) = diag(&S);
    let rootL_S = L_S.map(|x| if x > 0.0000001 {x.powf(-0.5)} else {0f64});
    let rootS = U_S * rootL_S * U_S.transpose();

    let F0 = rootS.transpose() * H * rootS;
    let (U_F0, L_F0) = diag(&F0);
    let mut C0 = rootS * U_F0;

    for i in 0..20 {
        let (E, C, e) = scf(C0, &H, &rootS, N, &ints);
        C0 = C;
        // if i  % 10 == 0 {
            println!("{} {}", i, E);
        // }
    }
}

fn scf(C: Mat77, H: &Mat77, rootS: &Mat77, N: usize, ints: &Integrals) -> (f64, Mat77, Mat77) {
    let mut D = Mat77::zeros();
    for u in 0..7 {
        for v in 0..7 {
            for i in 0..N/2 {
                D[u*7+v] += C[u*7+i] * C[v*7+i];
            }
        }
    }

    let mut F = H.clone();
    for u in 0..7 {
        for v in 0..7 {
            for p in 0..7 {
                for o in 0..7 {
                    F[u*7+v] += D[p*7+o] * (2.0*ints.two_e(u,v,p,o) - ints.two_e(u,p,v,o))
                }
            }
        }
    }

    let mut E = ints.Enuc;
    for u in 0..7 {
        for v in 0..7 {
            E += D[u*7+v] * (H[u*7+v] + F[u*7+v])
        }
    }

    let mut Fprime = rootS.transpose() * F * rootS;
    is_sy(&Fprime);
    diag_works(&Fprime);
    let (Cprime, epsilon) = diag(&Fprime);
    println!("{:?}\n{}\n{}", Fprime, Cprime, epsilon);

    return (E, rootS * Cprime, epsilon)
}

fn diag(F: &Mat77) -> (MatrixN<f64, U7>, MatrixN<f64, U7>) {
    let F0_diag = SymmetricEigen::new(F.clone());
    // the eigenvalues are unsorted, we should sort them and their corresponding eigenvectors
    let mut vs_F0 = F0_diag.eigenvectors;
    let mut ls_F0 = F0_diag.eigenvalues;
    println!("{}\n{}", vs_F0, ls_F0);

    let mut a = F.clone();
    let mut a = a.as_mut_slice();
    let mut w = vec![0f64; 7];
    let mut work = vec![0f64; 49];
    let lwork = work.len() as i32;
    let mut info = 0;

    unsafe {
        dsyev(b'V', b'U', 7, &mut a, 7, &mut w, &mut work, lwork, &mut info);
    }

    let b = Mat77::from_row_slice(a);
    println!("DSYEV\n{}", b);
    println!("{:?}", w);

    // bubble sort
    let mut elementsToCheck = ls_F0.len();
    let mut swaps = 1;
    while swaps > 0 {
        swaps = 0;
        for i in 0..elementsToCheck - 1 {
            if ls_F0[i] > ls_F0[i+1] {
                // swap these values
                let tmp = ls_F0[i+1];
                ls_F0[i+1] = ls_F0[i];
                ls_F0[i] = tmp;

                vs_F0.swap_columns(i,i+1);

                swaps += 1;
            }
        }

        elementsToCheck -= 1;
    }

    return (vs_F0, Mat77::from_diagonal(&ls_F0))
}

fn get_integrals(filename: &str) -> io::Result<Integrals> {
    let Enuc_re = Regex::new(r"Nuclear repulsion energy:\s+(\d+\.\d+)").unwrap();
    let type_re = Regex::new(r"##\s+([A-Za-z ]+)\s+##").unwrap();
    let int_re = Regex::new(r"\s*(\d)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*").unwrap();
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let lines = contents.lines().collect::<Vec<&str>>();

    let mut data = Integrals::default();
    for i in 0..7 {

    }

    for i in 0..lines.len() {
        match Enuc_re.captures(lines[i]) {
            Some(cap) => {
               data.Enuc = (&cap[1]).parse::<f64>().unwrap();
                break;
            },
            None => continue
        }
    };

    let mut last_line = 0;
    for i in 0..lines.len() {
        match type_re.captures(lines[i]) {
            Some(cap) => {
                let mut array = &mut data.overlap;
                match &cap[1] {
                    "Overlap" => (),
                    "Kinetic" => array = &mut data.kinetic,
                    "Potential" => array = &mut data.potential,
                    "One Electron Ints" => array = &mut data.one_electron,
                    _ => continue
                }

                for i in i+4..i+11 {
                    match int_re.captures(lines[i]) {
                        Some(cap) => {
                            //  this index starts at 1
                            let j = &cap[1].parse::<usize>().unwrap() - 1;
                            for k in 0..7 {
                                array[j][k] = (&cap[k+2]).parse::<f64>().unwrap();
                                last_line = i;
                            }
                        },
                        None => println!("hmm something went wrong on line {}", i+1)
                    }
                }

            },
            None => continue
        }
    }

    get_two_electron_integrals(&lines[last_line+1..], &mut data.two_electron);

    Ok(data)
}

fn get_two_electron_integrals(lines: &[&str], two_e: &mut [[[[f64; 7]; 7]; 7]; 7]) {
    let int_re = Regex::new(r"\s*\(\s*(\d)\s*(\d)[\s\|]*(\d)\s*(\d)\s*\)\s*=\s*(-?\d\.\d+)\s*").unwrap();
    for i in 0..lines.len() {
        match int_re.captures(lines[i]) {
            Some(cap) => {
                let u = (&cap[1]).parse::<usize>().unwrap();
                let v = (&cap[2]).parse::<usize>().unwrap();
                let p = (&cap[3]).parse::<usize>().unwrap();
                let o = (&cap[4]).parse::<usize>().unwrap();
                let e = (&cap[5]).parse::<f64>().unwrap();

                two_e[u][v][p][o] = e;
            },
            None => () // println!("hmm, something went wrong on line {}\n{}", i+1, lines[i])
        }
    }
}

fn is_sy(A: &Mat77) {
    for i in 0..7 {
        for j in 0..7 {
            assert!(A[i*7 + j] - A[j*7 + i] < 0.0000001);
        }
    }
}

fn diag_works(A: &Mat77) {
    let (P,D) = diag(&A);
    let B = P*D*P.transpose();
    for i in 0..49 {
        assert!(A[i] - B[i] < 0.5, "{} {}\n{}\n{}\n{}\n{}\n{:?}", A[i], B[i], A, B, P, D, A);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diag_works() {
        let mut A = Mat77::new_random();
        // make A symmetric
        for i in 0..7 {
            for j in 0..7 {
                A[j*7 + i] = A[i*7 + j]
            }
        }

        let (P,D) = diag(&A);
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
