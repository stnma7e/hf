#![allow(non_snake_case)]

use std::io;
use std::fs::File;
use std::io::prelude::*;
use regex::Regex;

#[derive(Default, Debug)]
pub struct Integrals {
    pub Enuc: f64,
    pub overlap: [[f64; 7]; 7],
    pub kinetic: [[f64; 7]; 7],
    pub potential: [[f64; 7]; 7],
    pub one_electron: [[f64; 7]; 7],
    pub two_electron: [[[[f64; 7]; 7]; 7]; 7]
}

impl Integrals {
    /// Retuns two-electron integral for the given orbitals. Should obey 8-fold symmetry:
    /// [pq|rs] = [qp|rs] = [pq|sr] = [qp|sr] = [rs|pq] = [sr|pq] = [rs|qp] = [sr|qp].
    /// Where p >= q, and r >= s. And the integrals are ordered with pq orbitals first if
    /// (p(p + 1)/2) + q > (r(r + 1)/2) + s, otherwise the rs orbitals are first.
    pub fn two_e(&self, mut u: usize, mut v: usize, mut p: usize, mut o: usize) -> f64 {
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

pub fn get_integrals(filename: &str) -> io::Result<Integrals> {
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

pub fn get_two_electron_integrals(lines: &[&str], two_e: &mut [[[[f64; 7]; 7]; 7]; 7]) {
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
