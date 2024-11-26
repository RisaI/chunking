use clap::Parser;
use rand::{distributions::Bernoulli, prelude::Distribution};
use rayon::prelude::*;
use rug::{ops::Pow, Integer, Rational};

fn c(m: u32, n: u32) -> Integer {
    Integer::from(m).binomial(n)
}

fn binomial_distribution(items: u32, errors: u32, error_rate: Rational) -> Rational {
    debug_assert!(errors <= items, "more errors than total items");
    debug_assert!(
        (0. ..=1.).contains(&error_rate),
        "error rate must be a probability [0, 1]"
    );

    c(items, errors)
        * error_rate.clone().pow(errors)
        * (Rational::ONE - error_rate).pow(items - errors)
}

fn num_ops(items: u32, chunks: u32, error_rate: Rational) -> Rational {
    fn num_ops_exact(items: u32, chunks: u32, nonempty_chunks: u32) -> u32 {
        chunks + nonempty_chunks * items / chunks
    }

    fn nonempty_chunk_probability(error_rate: Rational, chunk_size: u32) -> Rational {
        1 - (Rational::ONE - error_rate).pow(chunk_size)
    }

    debug_assert!(chunks > 0, "zero chunks not allowed");
    debug_assert!(chunks <= items, "more chunks than total items");
    debug_assert!(
        (0. ..=1.).contains(&error_rate),
        "error rate must be a probability [0, 1]"
    );

    let prob = |nonempty_chunks: u32| {
        debug_assert!(
            nonempty_chunks <= chunks,
            "more empty chunks than total chunks"
        );

        binomial_distribution(
            chunks,
            nonempty_chunks,
            nonempty_chunk_probability(error_rate.clone(), items / chunks),
        )
    };

    (0..=chunks)
        .into_par_iter()
        .map(|nonempty_chunks| {
            prob(nonempty_chunks) * num_ops_exact(items, chunks, nonempty_chunks)
        })
        .sum::<Rational>()
}

fn monte_carlo(
    items: u32,
    error_rate: Rational,
    chunk_size: usize,
    sample_size: usize,
) -> Rational {
    debug_assert!(
        chunk_size <= items as usize,
        "chunk is larger than total items"
    );
    debug_assert!(
        (0. ..=1.).contains(&error_rate),
        "error rate must be a probability [0, 1]"
    );

    let distr = Bernoulli::new(error_rate.to_f64()).unwrap();

    // Average op count
    Rational::from((
        (0..sample_size)
            .into_par_iter()
            .map(|_| {
                // Generate test data
                let test = distr
                    .sample_iter(rand::thread_rng())
                    .take(items as usize)
                    .collect::<Box<[bool]>>();

                // Calculate operations for each chunk
                test.chunks(chunk_size)
                    .map(|ch| {
                        Integer::from(if ch.iter().any(|v| *v) {
                            ch.len() + 1
                        } else {
                            1
                        })
                    })
                    .sum::<Integer>()
            })
            .sum::<Integer>(),
        sample_size as u32,
    ))
}

#[derive(clap::Parser)]
struct Options {
    items: u32,
    expected_errors: u32,

    #[clap(long)]
    mc_samples: Option<usize>,

    #[clap(long)]
    mc_only: bool,
}

fn main() {
    let Options {
        items,
        expected_errors,
        mc_samples,
        mc_only,
    } = Options::parse();

    let error_rate = Rational::from((expected_errors, items));

    for chunks in 1..items {
        if items % chunks != 0 {
            continue;
        }

        // Analytic solution
        if !mc_only {
            println!(
                "{: >4} chunks ({: >4} items/chunk) -> {:.1} ops by analytic method",
                chunks,
                items / chunks,
                num_ops(items, chunks, error_rate.clone()).to_f64()
            );
        }

        // Monte-Carlo solution
        if let Some(mc_samples) = mc_samples {
            println!(
                "{: >4} chunks ({: >4} items/chunk) -> {:.1} ops by monte carlo",
                chunks,
                items / chunks,
                monte_carlo(
                    items,
                    error_rate.clone(),
                    (items / chunks) as usize,
                    mc_samples
                )
                .to_f64()
            );
        }
    }
}
