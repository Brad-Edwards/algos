/// lib.rs

/// Computes the minimum number of coins needed to form the target `amount`.
///
/// The coin change problem here is the "unbounded" variant, meaning each coin
/// can be used any number of times. Returns `None` if it's impossible to form
/// the `amount` using the given `coins`.
///
/// # Examples
///
/// ```
/// use coinchange::min_coins_for_change;
///
/// // Minimum 3 coins: 6 + 6 + 6 = 18
/// let coins = vec![1, 6, 10];
/// assert_eq!(min_coins_for_change(&coins, 18), Some(3));
///
/// // Impossible to form 7 from [2,4], so returns None
/// let coins2 = vec![2, 4];
/// assert_eq!(min_coins_for_change(&coins2, 7), None);
/// ```
pub fn min_coins_for_change(coins: &[usize], amount: usize) -> Option<usize> {
    // If amount is 0, zero coins are needed.
    if amount == 0 {
        return Some(0);
    }
    // If no coins provided, can't form any positive amount.
    if coins.is_empty() {
        return None;
    }

    // dp[i] will hold the minimum number of coins to form amount i.
    // Initialize to a large number (here, use usize::MAX as sentinel).
    let mut dp = vec![usize::MAX; amount + 1];
    dp[0] = 0; // base case

    for &coin in coins {
        for curr_amount in coin..=amount {
            if dp[curr_amount - coin] != usize::MAX {
                dp[curr_amount] = dp[curr_amount].min(dp[curr_amount - coin] + 1);
            }
        }
    }

    if dp[amount] == usize::MAX {
        None
    } else {
        Some(dp[amount])
    }
}

/// Computes the number of distinct ways to form `amount` using the given `coins`.
///
/// This also uses an unbounded knapsack approach (each coin can be used
/// any number of times).
///
/// # Examples
///
/// ```
/// use coinchange::count_change_ways;
///
/// // There are 4 ways to make 5 using [1,2,5]:
/// //   1) 1+1+1+1+1
/// //   2) 1+1+1+2
/// //   3) 1+2+2
/// //   4) 5
/// assert_eq!(count_change_ways(&[1, 2, 5], 5), 4);
/// ```
pub fn count_change_ways(coins: &[usize], amount: usize) -> usize {
    // dp[i] will be the number of ways to form amount i.
    let mut dp = vec![0_usize; amount + 1];
    dp[0] = 1; // base case: 1 way to form 0 (use no coins)

    // We iterate over coins, then over amounts, so each coin
    // can be used multiple times in the correct unbounded manner.
    for &coin in coins {
        for curr_amount in coin..=amount {
            dp[curr_amount] += dp[curr_amount - coin];
        }
    }

    dp[amount]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_coins_for_change() {
        // Basic usage
        let coins = vec![1, 6, 10];
        assert_eq!(min_coins_for_change(&coins, 18), Some(3));
        assert_eq!(min_coins_for_change(&coins, 0), Some(0));
        assert_eq!(min_coins_for_change(&coins, 1), Some(1));

        // Impossible case
        let coins2 = vec![2, 4];
        assert_eq!(min_coins_for_change(&coins2, 7), None);
    }

    #[test]
    fn test_count_change_ways() {
        let coins = vec![1, 2, 5];
        assert_eq!(count_change_ways(&coins, 5), 4);

        // If amount is 0, there's exactly 1 way (use no coins).
        assert_eq!(count_change_ways(&coins, 0), 1);

        // Using [2,4], ways to form 8:
        //   - 2+2+2+2
        //   - 4+4
        // => 2 ways
        let coins2 = vec![2, 4];
        assert_eq!(count_change_ways(&coins2, 8), 2);

        // If coins is empty, 0 ways to form any positive amount
        assert_eq!(count_change_ways(&[], 5), 0);
    }
}
