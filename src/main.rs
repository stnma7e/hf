#![allow(non_snake_case)]

extern crate regex;
extern crate rand;
extern crate nalgebra as na;
extern crate lapack;

use std::io;
use std::fs::File;
use std::io::prelude::*;
use regex::Regex;

use lapack::fortran::*;
use na::{Matrix, MatrixN, VectorN, ArrayStorage, U7};

type Mat77 = Matrix<f64, U7, U7, ArrayStorage<f64, U7, U7>>;
type Vec7 = VectorN<f64, U7>;

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
    /// Retuns two-electron integral for the given orbitals. Should obey 8-fold symmetry:
    /// [pq|rs] = [qp|rs] = [pq|sr] = [qp|sr] = [rs|pq] = [sr|pq] = [rs|qp] = [sr|qp].
    /// Where p >= q, and r >= s. And the integrals are ordered with pq orbitals first if
    /// (p(p + 1)/2) + q > (r(r + 1)/2) + s, otherwise the rs orbitals are first.
    fn two_e(&self, mut u: usize, mut v: usize, mut p: usize, mut o: usize) -> f64 {
        // make sure u > v and p > o
        let mut tmp;
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
    let ints = get_integrals("h2oints.txt").unwrap();
    let S = Mat77::from_iterator(ints.overlap.iter().flatten().map(|i| i.clone()));
    let T = Mat77::from_iterator(ints.kinetic.iter().flatten().map(|i| i.clone()));
    let V = Mat77::from_iterator(ints.potential.iter().flatten().map(|i| i.clone()));
    let H = T + V;

    let (U_S, L_S) = diag(S);
    let rootL_S = L_S.map(|x| if x > 0.0000001 {x.powf(-0.5)} else {0f64});
    let rootS = U_S * rootL_S * U_S.transpose();

    let F0 = rootS.transpose() * H * rootS;
    let (U_F0, _) = diag(F0);
    let mut C0 = rootS * U_F0;
    let mut E_ref = 100000.0;

    for i in 0..2000 {
        let (E, C, _epsilon) = scf(C0, &H, &rootS, N, &ints);
        println!("{} {}", i, E);
        if E_ref - E < 10.0f64.powf(-6.0) {
            return;
        }

        E_ref = E;
        C0 = C;
    }
}

fn scf(C: Mat77, H: &Mat77, rootS: &Mat77, N: usize, ints: &Integrals) -> (f64, Mat77, Mat77) {
    let mut D = Mat77::zeros();
    for u in 0..7 {
        for v in 0..7 {
            for i in 0..N/2 {
                D[u+v*7] += C[u+i*7] * C[v+i*7];
            }
        }
    }

    let mut F = H.clone();
    for u in 0..7 {
        for v in 0..7 {
            for p in 0..7 {
                for o in 0..7 {
                    F[u+v*7] += D[p+o*7] * (2.0*ints.two_e(u,v,p,o) - ints.two_e(u,p,v,o))
                }
            }
        }
    }

    let mut E = ints.Enuc;
    for u in 0..7 {
        for v in 0..7 {
            E += D[u+v*7] * (H[u+v*7] + F[u+v*7])
        }
    }

    let Fprime = rootS.transpose() * F * rootS;
    let (Cprime, epsilon) = diag(Fprime);

    return (E, rootS * Cprime, epsilon)
}

fn diag(F: Mat77) -> (MatrixN<f64, U7>, MatrixN<f64, U7>) {
    let mut a = F;   // eigenvector output
    let mut w = Vec7::from_element(0f64); // eigenvalue output
    let mut work = vec![0f64; 49];
    let lwork = work.len() as i32;
    let mut info = 0;

    dsyev(b'V', b'U', 7, &mut a.as_mut_slice(), 7, w.as_mut(), &mut work, lwork, &mut info);

    return (a, Mat77::from_diagonal(&w))
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
