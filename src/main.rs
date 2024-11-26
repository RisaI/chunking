use rug::{ops::Pow, Integer, Rational};

fn c(m: u32, n: u32) -> Integer {
    Integer::from(m).binomial(n)
}

fn binomial(attempts: u32, successes: u32, probability: Rational) -> Rational {
    Rational::from(c(attempts, successes))
        * probability.clone().pow(successes)
        * (Rational::ONE - probability).pow(attempts - successes)
}

fn num_ops(items: u32, errors: u32, chunks: u32) -> Rational {
    let total_states = c(errors + chunks - 1, errors);

    let prob = |empty: u32| {
        let non_empty = chunks - empty;

        if non_empty > errors {
            return Rational::new();
        }

        c(chunks, non_empty)
            * (c(errors - 1, errors - non_empty))
            * Rational::from((1, total_states.clone()))
    };

    (0..chunks)
        .map(|j| prob(j) * (items - j * (Rational::from((items, chunks)) - 1)))
        .sum::<Rational>()
}

fn main() {
    let items = 3300;
    let error_rate = Rational::from((6, items));

    for chunks in 1..=items {
        if items % chunks != 0 {
            continue;
        }

        let ops = (0..=items)
            .map(|errs| binomial(items, errs, error_rate.clone()) * num_ops(items, errs, chunks))
            .sum::<Rational>();

        println!("{:?} chunks -> {:?} ops", chunks, ops.to_f64());
    }
}
