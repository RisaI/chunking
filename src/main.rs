use clap::Parser;
use rand::Rng;
use rayon::prelude::*;
use rug::{ops::Pow, Integer, Rational};

fn c(m: u32, n: u32) -> Integer {
    Integer::from(m).binomial(n)
}

fn binomial_distribution(attempts: u32, successes: u32, probability: Rational) -> Rational {
    c(attempts, successes)
        * probability.clone().pow(successes)
        * (Rational::ONE - probability).pow(attempts - successes)
}

fn num_ops_exact(items: u32, chunks: u32, free_chunks: u32) -> u32 {
    items + chunks - free_chunks * items / chunks
}

fn num_ops(items: u32, errors: u32, chunks: u32) -> Rational {
    let total_states = c(errors + chunks - 1, errors);

    let prob = |empty: u32| {
        let non_empty = chunks - empty;

        if non_empty > errors {
            return Rational::new();
        }

        Rational::from((
            c(chunks, non_empty) * (c(errors - 1, errors - non_empty)),
            total_states.clone(),
        ))
    };

    (0..chunks)
        .map(|free_chunks| (free_chunks, prob(free_chunks)))
        .skip_while(|(_, p)| p.is_zero())
        .map(|(free_chunks, p)| p * num_ops_exact(items, chunks, free_chunks))
        .sum::<Rational>()
}

fn monte_carlo(items: u32, error_rate: Rational, chunks: u32, sample_size: usize) -> Rational {
    let numer = error_rate.numer().to_u32().unwrap();
    let denom = error_rate.denom().to_u32().unwrap();

    // Average op count
    Rational::from((
        (0..sample_size)
            .into_par_iter()
            .map(|_| {
                // Generate test data
                let test = (0..items)
                    .map(|_| rand::thread_rng().gen_ratio(numer, denom))
                    .collect::<Box<_>>();

                // Calculate operations for each chunk
                test.chunks_exact((items / chunks) as usize)
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
}

fn main() {
    let Options {
        items,
        expected_errors,
        mc_samples,
    } = Options::parse();

    let error_rate = Rational::from((expected_errors, items));

    for chunks in 1..items {
        if items % chunks != 0 {
            continue;
        }

        // Analytic solution
        let ops = (0..=items)
            .into_par_iter()
            .map(|errs| {
                binomial_distribution(items, errs, error_rate.clone())
                    * num_ops(items, errs, chunks)
            })
            .sum::<Rational>();

        println!(
            "{: >4} chunks ({: >4} items/chunk) -> {:.1} ops",
            chunks,
            items / chunks,
            ops.to_f64()
        );

        // Monte-Carlo solution
        if let Some(mc_samples) = mc_samples {
            println!(
                "{: >4} chunks ({: >4} items/chunk) -> {:.1} ops by monte carlo",
                chunks,
                items / chunks,
                monte_carlo(items, error_rate.clone(), chunks, mc_samples).to_f64()
            );
        }
    }
}
